import torch
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
import nltk
from nltk import tokenize
from tools import Retriever
from prompts import retrieval_prompt
from typing import List
from data_generation.base_api import APICallPostprocessing

nltk.download("punkt")

# TODO: Per API?
MAX_BATCH_SIZE = 1  # My 3090 is weak 😔
N = 128  # SEQ Len
MAX_LEN = 1024  # Maximum retrieval length
M = 16  # Min Loss Span To Consider


class RetrievalPostprocessing(APICallPostprocessing):
    def __init__(
        self,
        start_tokens: List[int],
        end_tokens: List[int],
        minimum_percentage: float = 0.1,
    ):
        self.retriever = Retriever()
        self.api_text = "Retrieval("
        super().__init__(start_tokens, end_tokens, minimum_percentage)

    def add_api_calls(
        self,
        candidate: int,
        outputs: dict,
        texts_to_test: List[str],
        tokenizer: PreTrainedTokenizerBase,
        input_tokens: torch.Tensor,
        input_start: int,
        nums_to_keep: List[int],
        base_loss: float,
        *args,
        **kwargs
    ):
        retrieval_strings = args[0]
        generated_texts = list()
        max_token_len = N
        max_token_len_base = N
        for j in range(len(outputs)):
            outputs[j]["Retrieval"] = outputs[j]["generated_text"].replace(
                texts_to_test[candidate], ""
            )
            outputs[j]["Generated"] = outputs[j]["generated_text"].split("Output:")[-1]
            if "]" in outputs[j]["Retrieval"]:
                outputs[j]["Retrieval"] = (
                    outputs[j]["Retrieval"].replace("Retrieval(", "").split("]")[0]
                )
                if ")" in outputs[j]["Retrieval"]:
                    outputs[j]["Retrieval"] = outputs[j]["Retrieval"].split(")")[0]
                outputs[j]["Retrieval_text"] = (
                    "[Retrieval(" + outputs[j]["Retrieval"] + ")"
                )
                base_inputs = tokenizer(
                    outputs[j]["Retrieval_text"] + "]" + "\n",
                    return_tensors="pt",
                )["input_ids"].cuda()
                outputs[j]["Retrieval"] = self.retriever.retrieval(
                    retrieval_strings, outputs[j]["Retrieval"], 3
                )
                outputs[j]["Retrieval_output"] = [outputs[j]["Retrieval_text"][1:], ", ".join(outputs[j]["Retrieval"])]
                outputs[j]["Retrieval_text"] = (
                    outputs[j]["Retrieval_text"]
                    + "->"
                    + ", ".join(outputs[j]["Retrieval"])
                    + "]"
                )
                test_inputs = tokenizer(
                    outputs[j]["Retrieval_text"] + "\n",
                    return_tensors="pt",
                )["input_ids"].cuda()
                test_inputs = torch.concat(
                    [
                        test_inputs.cuda(),
                        input_tokens[:, input_start:].cuda(),
                    ],
                    dim=1,
                )
                if test_inputs.shape[1] > MAX_LEN:
                    continue
                base_inputs = torch.concat(
                    [
                        base_inputs.cuda(),
                        input_tokens[:, input_start:].cuda(),
                    ],
                    dim=1,
                )
                max_token_len = max(max_token_len, test_inputs.shape[1])
                max_token_len_base = max(max_token_len_base, test_inputs.shape[1])
                generated_texts.append(
                    [
                        test_inputs,
                        base_inputs,
                        nums_to_keep[candidate],
                        base_loss,
                        outputs[j],
                    ]
                )
        return generated_texts, max_token_len, max_token_len_base

    def parse_article(
        self, data: dict, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
    ):
        outputs = list()
        tokens = tokenizer(data["text"], return_tensors="pt")["input_ids"]
        start_step = 2048//N
        ret_skip = 1024//N  # naively assuming the model should be able to look back if it's less than this.
        total_steps = tokens.shape[1]//N
        for i in range(start_step, total_steps):
            input_tokens = tokens[:, (-N * (i + 1) - 1) : (-N * (i) - 1)]
            labels = tokens[
                :,
                int(tokens.shape[1] + (-N * (i + 1))) : int(tokens.shape[1] + (-N * i)),
            ]
            ret_tokens = tokens[:, : (-(N) * ((i - ret_skip) + 1) - 1)]
            # print(tokens.shape)
            string = tokenizer.decode(input_tokens[0])
            ret_strings = tokenize.sent_tokenize(tokenizer.decode(ret_tokens[0]))
            # print(ret_strings)
            model_input = tokenizer(
                retrieval_prompt.replace("<REPLACEGPT>", string) + string,
                return_tensors="pt",
            )["input_ids"]
            # print(string)
            # print(model_input.shape)
            with torch.no_grad():
                output = model(model_input.cuda()).logits.cpu()[:, -N:]
            new_outputs = self.generate_continuations(
                model_input,
                output,
                labels,
                model,
                tokenizer,
                ret_strings,
            )
            for output in new_outputs:
                if output is None:
                    continue
                output["index"] += int(tokens.shape[1] + (-N * (i + 1)))
                # filter by score
                if output["Score"] > 1.0:
                    outputs.append([output["Score"], output["index"]] + output["Retrieval_output"])
        return outputs
