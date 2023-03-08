import argparse
import asyncio
import re
from typing import Callable, Dict, Iterator, Optional, Tuple, Awaitable

import torch
import torch.nn.functional as F
import tritonclient.grpc.aio as grpcclient
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from tools import Calculator, Tool, WikiSearch

"""
Tool use parsing regex
[Calculator(1 / 2) -> 0.5] into ("Calculator", "400 / 1400", "0.5")
[Date()] into ("Date", None, None)
[WikiSearch('abcdef')] into ("WikiSearch", "'abcdef'", None)
"""
TOOL_REGEX = "\[([A-Za-z]+)\((.*)\)(\s->\s.*)?\]"


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
    api_call_threshold: float = 0.1,
    max_length: int = 1024,
    new_tokens: int = 100,
):

    prompted_text = tool.prompt.replace("<REPLACEGPT>", prompt) + prompt
    prompted_tokens = tokenizer(prompted_text, return_tensors="pt", truncation=True, max_length=max_length - new_tokens).input_ids
    prefix_tokens = tokenizer(
        tool.prompt.replace("<REPLACEGPT>", prompt),
        return_tensors="pt",
        truncation=True,
    ).input_ids
    logits, _ = await model(prompted_tokens)
    probs = F.softmax(logits.float(), dim=-1) # no CPU softmax for half

    # Find top k locations with highest probability of <API> tokens
    probs_for_api = probs[0, prefix_tokens.shape[1] :, start_tokens].max(dim=-1).values
    top_k_probs, top_k_indices = torch.topk(probs_for_api, min(len(probs_for_api), k))

    for idx, prob in zip(top_k_indices, top_k_probs):
        if prob.item() < api_call_threshold:
            break

        continuation_start = idx.item() + prefix_tokens.shape[1]
        print(f"{continuation_start=}")

        _, api_calls = await model(
            prompted_tokens[:, :continuation_start],
            new_tokens=new_tokens
        )

        for api_call in api_calls:
            api_call_str = tokenizer.decode(api_call[continuation_start:])
            api_call_str = tokenizer.decode(api_call)
            print(f"{api_call_str=}")
            match = re.match(TOOL_REGEX, api_call_str)
            if match is None:
                continue
            tool_name = match.group(1)
            tool_args = match.group(2)

            if tool_name == tool.__class__.__name__:
                api_output_str = tool(tool_args)

                yield tool_name, tool_args, api_output_str


async def api_loss_reduction(
    model: Callable[[torch.Tensor], Awaitable[torch.Tensor]],
    tokenizer: AutoTokenizer,
    tool_name,
    tool_args,
    tool_output,
    prompt,
    weights=None,
):
    with_api = f"[{tool_name}({tool_args}) -> {tool_output}]{prompt}"
    with_args = f"[{tool_name}({tool_args})]{prompt}"
    without_api = f"{prompt}"

    prompt_tokens = tokenizer([with_api, with_args, without_api], return_tensors="pt")[
        "input_ids"
    ]
    logits, _ = await model(prompt_tokens)
    losses = F.cross_entropy(
        logits, prompt_tokens, reduction="none", weight=weights
    )
    L_plus = losses[0].sum()
    L_minus = min(losses[1].sum(), losses[2].sum())
    return L_minus - L_plus


def prepare_inference_inputs(inputs_ids: torch.IntTensor, new_tokens: int = 1, temperature: float = 1.0):
    batch_size = inputs_ids.shape[0]

    input_ids_input = grpcclient.InferInput("input_ids", inputs_ids.shape, "INT32")  
    input_ids_input.set_data_from_numpy(inputs_ids.int().cpu().numpy())

    new_tokens_input = grpcclient.InferInput("tensor_of_seq_len", [batch_size, new_tokens], "INT32")
    new_tokens_input.set_data_from_numpy(torch.zeros(batch_size, new_tokens, dtype=torch.int32).cpu().numpy())

    temperature_input = grpcclient.InferInput("temperature", [batch_size, 1], "FP32")
    temperature_input.set_data_from_numpy(torch.full([batch_size, 1], temperature, dtype=torch.float32).cpu().numpy())

    inputs = [input_ids_input, new_tokens_input, temperature_input]
    outputs = [grpcclient.InferRequestedOutput("logits"), grpcclient.InferRequestedOutput("output_ids")]
    return inputs, outputs


async def infer(triton_client, model_name, input_ids, new_tokens: int = 1, temperature: float = 1.0):
    inputs, outputs = prepare_inference_inputs(input_ids, new_tokens, temperature)
    result = await triton_client.infer(
        model_name=model_name, inputs=inputs, outputs=outputs
    )
    logits = torch.tensor(result.as_numpy("logits").copy(), requires_grad=False)
    output_ids = torch.tensor(result.as_numpy("output_ids").copy(), requires_grad=False)
    return logits, output_ids

async def main(model_name, url):
    async with grpcclient.InferenceServerClient(
        url=url,
    ) as triton_client:

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.truncation_side = "left"

        async def infer_model(input_ids, new_tokens: int = 1, temperature: float = 1.0):
            return await infer(triton_client, model_name, input_ids, new_tokens, temperature)
        
        start_tokens = [
            tokenizer("[")["input_ids"][0],
            tokenizer(" [")["input_ids"][0],
        ]
        end_token = tokenizer("]")["input_ids"][0]

        tools = [Calculator(), WikiSearch()]

        dataset = load_dataset("c4", "en", split="train", streaming=True)
        iter_data = iter(dataset)

        counter = 0
        while counter < 10:
            data = next(iter_data)
            for tool in tools:
                if tool.heuristic(data):
                    async for tool_name, tool_args, tool_output in sample_api_calls(
                        # assumes tool's name is the same as the name of the class implementing it
                        tool,
                        infer_model,
                        tokenizer,
                        data["text"],
                        5,
                        5,
                        start_tokens,
                        end_token,
                        0.1,
                    ):
                        print(tool_name, tool_args, tool_output)
                        print(
                            api_loss_reduction(
                                infer_model,
                                tokenizer,
                                tool_name,
                                tool_args,
                                tool_output,
                                data["text"],
                            )
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Name of the model to use",
    )

    parser.add_argument(
        "--url",
        type=str,
        default="localhost:8001",
        help="URL to the GRPCInferenceService of Triton Inference Server",
    )

    args = parser.parse_args()
    asyncio.run(main(args.model_name, args.url))
