import json

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    AutoTokenizer,
    PreTrainedModel,
    pipeline,
)
from datasets import load_dataset
from dataclasses import dataclass
import nltk
from nltk import tokenize
from tools import Retriever
from prompts import retrieval_prompt
from typing import List

nltk.download("punkt")

# TODO: Per API?
MAX_BATCH_SIZE = 1  # My 3090 is weak 😔
N = 64  # SEQ Len
M = 16  # Min Loss Span To Consider


@dataclass
class AvailableAPIs:
    """Keeps track of available APIs"""

    retrieval: bool = True

    def check_any_available(self):
        return any([self.retrieval])


class APICallPostprocessing:
    def __init__(
        self,
        start_tokens: List[int],
        end_tokens: List[int],
        minimum_percentage: float = 0.1,
    ):
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens
        self.minimum_percentage = minimum_percentage
        self.retriever = Retriever()

    def find_and_rank(
        self,
        input_tokens: torch.Tensor,
        input_logits: torch.Tensor,
        labels: torch.Tensor,
        api_text: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        retrieval_strings: List[str],
    ):
        # First, figure out locations...
        input_start = input_tokens.shape[1] - input_logits.shape[1]
        start_str = tokenizer.decode(input_tokens[:, :input_start][0])
        probs = torch.softmax(input_logits, dim=-1)
        remove_tokens = 1.0 - torch.sum(
            torch.stack([labels == start_token for start_token in self.start_tokens]),
            dim=0,
        )
        # print(remove_tokens)
        max_start_tokens = torch.amax(
            torch.stack(
                [probs[:, :, start_token] for start_token in self.start_tokens]
            ),
            dim=0,
        )
        # remove tokens where it's appropriate to be the start token, e.g. citations maybe?
        max_start_tokens = max_start_tokens * remove_tokens
        # Each sequence find top 5...
        values, indicies = torch.topk(max_start_tokens[:, : -(M + 1)], k=5, dim=1)
        # setup generation calls...
        generator = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, device=0
        )
        criterion = nn.CrossEntropyLoss()
        generated_texts = list()
        texts_to_test = list()
        max_index = 0
        outputs = list()
        num_to_keeps = list()
        with torch.no_grad():
            for i, batch in enumerate(indicies):
                for j, index in enumerate(batch):
                    if values[i][j] < self.minimum_percentage:
                        continue
                    base_outputs = model(input_tokens[:, input_start:].cuda()).logits[
                        :, index:
                    ]
                    print(base_outputs.shape)
                    num_keep = int(base_outputs.shape[1])
                    base_loss = criterion(
                        base_outputs.view(-1, base_outputs.size(-1)),
                        labels[:, index:].cuda().view(-1),
                    )
                    max_index = max(max_index, index)
                    texts_to_test.append(
                        tokenizer.decode(input_tokens[:, : input_start + index][i])
                        + f" [{api_text}"
                    )
                    outputs.append(
                        generator(
                            texts_to_test[-1], max_new_tokens=28, num_return_sequences=5
                        )
                    )
                    for k in range(5):
                        outputs[-1][k]["index"] = int(index)
                    num_to_keeps.append(num_keep)
            for i in range(len(outputs)):
                generated_texts = list()
                max_token_len = N
                for j in range(len(outputs[i])):
                    outputs[i][j]["Retrieval"] = outputs[i][j][
                        "generated_text"
                    ].replace(texts_to_test[i], "")
                    outputs[i][j]["Generated"] = outputs[i][j]["generated_text"].split(
                        "Output:"
                    )[-1]
                    if "]" in outputs[i][j]["Retrieval"]:
                        outputs[i][j]["Retrieval"] = outputs[i][j]["Retrieval"].split(
                            "]"
                        )[0]
                        if ")" in outputs[i][j]["Retrieval"]:
                            outputs[i][j]["Retrieval"] = outputs[i][j][
                                "Retrieval"
                            ].split(")")[0]
                        outputs[i][j]["Retrieval_text"] = (
                            "[Retrieval(" + outputs[i][j]["Retrieval"] + ")]"
                        )
                        outputs[i][j]["Retrieval"] = self.retriever.retrieval(
                            retrieval_strings, outputs[i][j]["Retrieval"], 3
                        )
                        outputs[i][j]["Retrieval_text"] = (
                            outputs[i][j]["Retrieval_text"]
                            + "->["
                            + ", ".join(outputs[i][j]["Retrieval"])
                            + "]"
                        )
                        test_inputs = tokenizer(
                            "Retrieved: "
                            + ", ".join(outputs[i][j]["Retrieval"])
                            + "\n",
                            return_tensors="pt",
                        )["input_ids"].cuda()
                        test_inputs = torch.concat(
                            [
                                test_inputs.cuda(),
                                input_tokens[:, input_start:].cuda(),
                            ],
                            dim=1,
                        )
                        max_token_len = max(max_token_len, test_inputs.shape[1])
                        generated_texts.append(
                            [test_inputs, num_keep, base_loss, outputs[i][j]]
                        )
                # shape the batches...
                for j in range(len(generated_texts)):
                    generated_texts[j].append(
                        max_token_len - generated_texts[j][0].shape[1]
                    )
                    if generated_texts[j][-1] != 0:
                        generated_texts[j][0] = torch.cat(
                            (
                                generated_texts[j][0],
                                torch.zeros(
                                    (1, generated_texts[j][-1]),
                                    dtype=generated_texts[j][0].dtype,
                                    device=generated_texts[j][0].device,
                                ),
                            ),
                            dim=1,
                        )

                test_outputs = model(
                    torch.cat(
                        list(generated_text[0] for generated_text in generated_texts),
                        dim=0,
                    )
                ).logits
                best_loss = -99.0
                best_output = outputs[i][0]
                for j in range(len(generated_texts)):
                    if generated_texts[j][-1] != 0:
                        test = test_outputs[j][: -generated_texts[j][-1]]
                        test_loss = criterion(
                            test[-num_to_keeps[i] :].view(-1, base_outputs.size(-1)),
                            labels[:, -num_to_keeps[i] :].cuda().view(-1),
                        )
                    else:
                        test_loss = criterion(
                            test_outputs[j][-num_to_keeps[i] :].view(
                                -1, base_outputs.size(-1)
                            ),
                            labels[:, -num_to_keeps[i] :].cuda().view(-1),
                        )
                    generated_texts[j][-2]["generated_text"] = generated_texts[j][-2][
                        "generated_text"
                    ].replace(start_str, "")
                    if base_loss - test_loss > best_loss:
                        best_output = generated_texts[j][-2]
                        best_loss = base_loss - test_loss
                outputs[i] = best_output
                outputs[i]["Score"] = float(best_loss.item())
        return outputs


def check_apis_available(
    data: dict, tokenizer: PreTrainedTokenizerBase
) -> AvailableAPIs:
    tokenized_data = tokenizer(data["text"])["input_ids"]
    available = AvailableAPIs()
    if len(tokenized_data) < 8000:
        available.retrieval = False
    return available


if __name__ == "__main__":
    gpt_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    prompt_tokens = gpt_tokenizer(retrieval_prompt, return_tensors="pt")["input_ids"]
    start_tokens = [
        gpt_tokenizer("[")["input_ids"][0],
        gpt_tokenizer(" [")["input_ids"][0],
    ]
    end_tokens = [
        gpt_tokenizer("]")["input_ids"][0],
        gpt_tokenizer(" ]")["input_ids"][0],
    ]  # TODO: keep second?
    api_handler = APICallPostprocessing(start_tokens, end_tokens)
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).cuda()
    dataset = load_dataset("c4", "en", split="train", streaming=True)
    iter_data = iter(dataset)
    test = False
    while not test:
        data = next(iter_data)
        available = check_apis_available(data, gpt_tokenizer)
        test = available.retrieval
        if test:
            # print(data)
            for i in range(5):
                tokens = gpt_tokenizer(data["text"], return_tensors="pt")["input_ids"]
                input_tokens = tokens[:, (-N * (i + 1) - 1) : (-N * (i) - 1)]
                labels = tokens[
                    :,
                    int(tokens.shape[1] + (-N * (i + 1))) : int(
                        tokens.shape[1] + (-N * i)
                    ),
                ]
                ret_tokens = tokens[:, : (-N * (i + 1) - 1)]
                print(tokens.shape)
                string = gpt_tokenizer.decode(input_tokens[0])
                ret_strings = tokenize.sent_tokenize(
                    gpt_tokenizer.decode(ret_tokens[0])
                )
                # print(ret_strings)
                model_input = gpt_tokenizer(
                    retrieval_prompt.replace("<REPLACEGPT>", string) + string,
                    return_tensors="pt",
                )["input_ids"]
                print(string)
                print(model_input.shape)
                with torch.no_grad():
                    output = model(model_input.cuda()).logits.cpu()[:, -N:]
                api_handler.find_and_rank(
                    model_input,
                    output,
                    labels,
                    "Retrieval(",
                    model,
                    gpt_tokenizer,
                    ret_strings,
                )
