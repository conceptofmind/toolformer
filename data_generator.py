import argparse
import asyncio
import json
import re

from dataclasses import dataclass
from itertools import islice
from typing import Callable, Dict, Iterator, Optional, Tuple, Awaitable

import torch
import torch.nn.functional as F
import tritonclient.grpc.aio as grpcclient
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from tools import Calculator, Tool, WikiSearch

"""
Tool use parsing regex
[Calculator(1 / 2) -> 0.5] into ("Calculator", "400 / 1400", "0.5")
[Date()] into ("Date", None, None)
 [WikiSearch('abcdef')] into ("WikiSearch", "'abcdef'", None)
"""
TOOL_REGEX = "\s?\[([A-Za-z]+)\((.*)\)(\s->\s.*)?\]"

TOOLFORMER_API_START = "<TOOLFORMER_API_START>"
TOOLFORMER_API_RESPONSE = "<TOOLFORMER_API_RESPONSE>"
TOOLFORMER_API_END = "<TOOLFORMER_API_END>"


@dataclass
class ToolUse:
    tool: Tool
    args: str
    output: str
    tokens: torch.Tensor
    prompt: str
    insert_index: int

    def __str__(self):
        return f"{self.tool.name}({self.args}) -> {self.output}"

    def render(self, tokenizer: AutoTokenizer) -> str:
        """
        Renders the tool use as a string.
        """
        prefix = tokenizer.decode(self.tokens[0, : self.insert_index])
        suffix = tokenizer.decode(self.tokens[0, self.insert_index :])
        return f"{prefix}{TOOLFORMER_API_START}{self.tool.name}({self.args}){TOOLFORMER_API_RESPONSE}{self.output}{TOOLFORMER_API_END}{suffix}"


def interpret_tools(input_text: str, tools: Dict[str, Callable[[str], str]]) -> str:
    """
    Interprets the use of tools in text and replaces them with their output.
    """
    output_text = input_text
    for match in re.finditer(TOOL_REGEX, input_text):
        tool_name = match.group(1)
        tool_args = match.group(2)  # empty string if no args
        tool_output: Optional[str] = match.group(3)
        if tool_name in tools and tool_output is None:
            args = [tool_args] if tool_args else []
            tool_output = tools[tool_name](*args)
            output = f"[{tool_name}({tool_args}) -> {tool_output}]"
            output_text = output_text.replace(match.group(0), output)
        elif tool_name not in tools:
            print(f"Unknown tool: {tool_name}")

    return output_text


"""
To sample API calls, we write a prompt that encourages a LM to annotate text with API calls.
Then, we find the top k locations with the highest probability of <API> tokens.
Then, we sample m API calls from the top k locations, giving the prompt <API> as a prefix and </API> as end of sentence token.

Then we execute all found API calls and keep their results.
Then we filter API calls by measuring the cross entropy loss between the original text with API call and results prefixed to it, 
the original text with no call, and the original text with the API call args but without outputs.
"""


async def sample_api_calls(
    tool: Tool,
    model: Callable[[torch.Tensor], Awaitable[Tuple[torch.Tensor, torch.Tensor]]],
    tokenizer: AutoTokenizer,
    prompt: str,
    k: int,
    m: int,
    start_tokens: int,
    end_token: int,
    api_call_threshold: float = 0.05,
    max_length: int = 1024,
    new_tokens: int = 100,
):

    # Build annotator prompt with <REPLACEGPT> as placeholder for the prompt
    # like "Input: example1\nOutput: annotated example1\nInput: <REPLACEGPT>\nOutput: "
    annotate_prompt = tool.prompt.replace("<REPLACEGPT>", prompt)
    # Concate prompt to `...\nOutput: `
    unannotated_output = annotate_prompt + prompt

    prompted_tokens = tokenizer(
        [unannotated_output],
        return_tensors="pt",
        truncation=True,
        max_length=max_length - new_tokens,
    ).input_ids
    prefix_tokens = tokenizer(
        [annotate_prompt],
        return_tensors="pt",
        truncation=True,
        max_length=max_length - new_tokens,
    ).input_ids
    input_tokens = prompted_tokens[:, :-1]
    labels = prompted_tokens[:, 1:]
    logits, _ = await model(input_tokens)
    probs = F.softmax(logits.float(), dim=-1)

    # Find top k locations with highest probability of <API> tokens

    # Make sure we don't keep any tokens that are supposed to be [ or try to
    # insert tokens info the prefix
    remove_tokens = ~torch.any(
        torch.stack([labels == start_token for start_token in start_tokens]),
        dim=0,
    )
    remove_tokens[:, : prefix_tokens.shape[1]] = False

    probs_for_api = probs[0, :, start_tokens].max(dim=-1).values
    probs_for_api = probs_for_api * remove_tokens.float()

    top_k_probs, top_k_indices = torch.topk(probs_for_api, min(len(probs_for_api), k))

    for idx, prob in zip(top_k_indices, top_k_probs):
        if prob.item() < api_call_threshold:
            break

        insert_api_at = idx.item()

        selected_start_tokens = probs[:, insert_api_at, start_tokens].argmax().item()

        for i in range(m):
            _, api_calls = await model(
                torch.cat(
                    [
                        input_tokens[:, :insert_api_at],
                        torch.full(
                            [1, 1],
                            start_tokens[selected_start_tokens],
                            dtype=torch.long,
                        ),
                    ],
                    dim=1,
                ),
                new_tokens=new_tokens,
            )

            api_call_str = tokenizer.decode(api_calls[0][insert_api_at:])
            match = re.match(TOOL_REGEX, api_call_str)

            if match is None:
                continue

            tool_name = match.group(1)
            tool_args = match.group(2)

            if tool_name == tool.name:
                api_output = tool(tool_args)
                few_shot_prompt_start = prefix_tokens.shape[1]
                prompt_tokens = input_tokens[:, few_shot_prompt_start:]
                yield ToolUse(
                    tool=tool,
                    args=tool_args,
                    output=api_output,
                    insert_index=insert_api_at - few_shot_prompt_start,
                    prompt=prompt,
                    tokens=prompt_tokens,
                )


async def api_loss_reduction(
    model: Callable[[torch.Tensor], Awaitable[torch.Tensor]],
    tokenizer: AutoTokenizer,
    tool_use: ToolUse,  
    max_length: int = 1024,
):
    api_use = tokenizer(
        f" [{tool_use.tool.name}({tool_use.args}) -> {tool_use.output}]",
        return_tensors="pt",
    )

    api_args = tokenizer(
        f"[{tool_use.tool.name}({tool_use.args})]",
        return_tensors="pt",
    )

    input_list = [
        torch.cat([api_use.input_ids[0], tool_use.tokens[0]], dim=0)[:max_length],
        torch.cat([api_args.input_ids[0], tool_use.tokens[0]], dim=0)[:max_length],
        tool_use.tokens[0],
    ]

    inputs = torch.nn.utils.rnn.pad_sequence(
        input_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )[:, :max_length]

    input_tokens = inputs[:, :-1]
    label_tokens = inputs[:, 1:]
    input_lengths = torch.tensor([x.shape[0] for x in input_list])
    prompt_length = input_lengths[2].item() - 1
    suffix_length = prompt_length - tool_use.insert_index

    logits, _ = await model(input_tokens)

    def weighted_cross_entropy(logits, labels, length):
        un_weighted_xent = F.cross_entropy(
            logits[length - suffix_length : length],
            labels[length - suffix_length : length],
            reduction="none",
        )

        weights = 1.0 - 0.2 * torch.arange(un_weighted_xent.shape[0])
        weights = torch.maximum(weights, torch.zeros_like(weights))
        return (un_weighted_xent * weights).sum()

    L_plus = weighted_cross_entropy(logits[0], label_tokens[0], input_lengths[0])

    L_minus_with_api_args = weighted_cross_entropy(
        logits[1], label_tokens[1], input_lengths[1]
    )

    L_minus_without_api = weighted_cross_entropy(
        logits[2], label_tokens[2], input_lengths[2]
    )

    L_minus = min(L_minus_with_api_args, L_minus_without_api)

    return L_minus - L_plus


def prepare_inference_inputs(
    inputs_ids: torch.IntTensor, new_tokens: int = 1, temperature: float = 1.0
):
    batch_size = inputs_ids.shape[0]

    input_ids_input = grpcclient.InferInput("input_ids", inputs_ids.shape, "INT32")
    input_ids_input.set_data_from_numpy(inputs_ids.int().cpu().numpy())

    new_tokens_input = grpcclient.InferInput(
        "tensor_of_seq_len", [batch_size, new_tokens], "INT32"
    )
    new_tokens_input.set_data_from_numpy(
        torch.zeros(batch_size, new_tokens, dtype=torch.int32).cpu().numpy()
    )

    temperature_input = grpcclient.InferInput("temperature", [batch_size, 1], "FP32")
    temperature_input.set_data_from_numpy(
        torch.full([batch_size, 1], temperature, dtype=torch.float32).cpu().numpy()
    )

    inputs = [input_ids_input, new_tokens_input, temperature_input]
    outputs = [
        grpcclient.InferRequestedOutput("logits"),
        grpcclient.InferRequestedOutput("output_ids"),
    ]
    return inputs, outputs


async def infer(
    triton_client, model_name, input_ids, new_tokens: int = 1, temperature: float = 1.0
):

    inputs, outputs = prepare_inference_inputs(input_ids, new_tokens, temperature)

    triton_model_name = model_name.replace("/", "--")

    result = await triton_client.infer(
        model_name=triton_model_name, inputs=inputs, outputs=outputs
    )

    logits = torch.tensor(result.as_numpy("logits").copy(), requires_grad=False)
    output_ids = torch.tensor(result.as_numpy("output_ids").copy(), requires_grad=False)

    return logits, output_ids


async def main(
    model_name,
    url,
    output_file,
    tau=0.5,
    max_concurrent=32,
    max_samples=10,
    max_datapoints=None,
    max_length=1024,
):
    async with grpcclient.InferenceServerClient(
        url=url,
    ) as triton_client:

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.truncation_side = "left"

        print(
            "Example output:",
            ToolUse(
                tool=Calculator(),
                args="1+1",
                output="2",
                tokens=tokenizer("the result of 1+1 is 2", return_tensors="pt").input_ids,
                prompt="the result of 1+1 is 2",
                insert_index=6,
            ).render(tokenizer=tokenizer),
        )

        async def infer_model(input_ids, new_tokens: int = 1, temperature: float = 1.0):
            return await infer(
                triton_client, model_name, input_ids, new_tokens, temperature
            )

        start_tokens = [
            tokenizer("[")["input_ids"][0],
            tokenizer(" [")["input_ids"][0],
        ]
        end_token = tokenizer("]")["input_ids"][0]

        tools = [Calculator(), WikiSearch()]

        dataset = load_dataset("c4", "en", split="train", streaming=True)
        iter_data = iter(dataset)

        if max_datapoints is not None:
            iter_data = islice(iter_data, max_datapoints)

        async def sample_and_filter_api_calls(tool, text, top_k, n_gen):
            async for tool_use in sample_api_calls(
                tool=tool,
                model=infer_model,
                tokenizer=tokenizer,
                prompt=text,
                k=top_k,
                m=n_gen,
                start_tokens=start_tokens,
                end_token=end_token,
                max_length=max_length,
                api_call_threshold=0.05,
            ):
                lm_loss_diff = await api_loss_reduction(
                    infer_model, tokenizer, tool_use, max_length=max_length
                )
                if lm_loss_diff.item() > tau:
                    return tool_use

        async def sample_tool_use(data):
            for tool in tools:
                if tool.heuristic(data):
                    return await sample_and_filter_api_calls(
                        tool, data["text"], top_k=5, n_gen=1
                    )

        pbar = tqdm(total=max_datapoints)
        pbar.set_description("Datapoints processed")

        tooled_pbar = tqdm(total=max_samples)
        tooled_pbar.set_description("Tool uses sampled")
        with open(output_file, "w") as f:
            counter = 0

            while True:
                data_samples = [next(iter_data) for _ in range(max_concurrent)]
                tasks = [
                    asyncio.create_task(sample_tool_use(data)) for data in data_samples
                ]
                pbar.update(len(data_samples))

                for sampled_tool_use in asyncio.as_completed(tasks):
                    tool_use = await sampled_tool_use   
                    if tool_use is not None:
                        counter += 1
                        tooled_pbar.update(1)
                        print(tool_use)

                        f.write(
                            json.dumps(dict(text=tool_use.render(tokenizer))) + "\n"
                        )
                        f.flush()

                        if counter > max_samples:
                            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/gpt-j-6B",
        help="Name of the model to use",
    )

    parser.add_argument(
        "--url",
        type=str,
        default="localhost:8001",
        help="URL to the GRPCInferenceService of Triton Inference Server",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="output.jsonl",
        help="Path to the output file",
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="Threshold for LM loss reduction the API call to be considered useful",
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=32,
        help="Maximum number of samples to process concurrently",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum number of tool annotated samples to generate",
    )

    parser.add_argument(
        "--max-datapoints",
        type=int,
        default=None,
        help="Maximum number of datapoints to sample from",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum length in tokens of the generated text",
    )

    args = parser.parse_args()
    asyncio.run(
        main(
            args.model_name,
            args.url,
            args.output_file,
            args.tau,
            args.max_concurrent,
            args.max_samples,
            args.max_datapoints,
            args.max_length,
        )
    )
