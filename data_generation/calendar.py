import torch
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from tools import Calendar
from prompts import calendar_prompt
from typing import List
from data_generation.base_api import APICallPostprocessing
import dateutil.parser as dparser


# TODO: Per API?
MAX_BATCH_SIZE = 1  # My 3090 is weak 😔
N = 128  # SEQ Len
M = 16  # Min Loss Span To Consider
MAX_LEN = 1024  # Maximum calendar length


class CalendarPostprocessing(APICallPostprocessing):
    def __init__(
        self,
        start_tokens: List[int],
        end_tokens: List[int],
        minimum_percentage: float = 0.0,
    ):
        self.calendar = Calendar
        self.api_text = "Calendar("
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
        calendar_string = args[0]
        generated_texts = list()
        max_token_len = N
        max_token_len_base = N
        for j in range(len(outputs)):
            outputs[j]["Calendar"] = outputs[j]["generated_text"].replace(
                texts_to_test[candidate], ""
            )
            outputs[j]["Generated"] = outputs[j]["generated_text"].split("Output:")[-1]
            if "]" in outputs[j]["Calendar"]:
                outputs[j]["Calendar"] = (
                    outputs[j]["Calendar"].replace("Calendar(", "").split("]")[0]
                )
                if ")" in outputs[j]["Calendar"]:
                    outputs[j]["Calendar"] = outputs[j]["Calendar"].split(")")[0]
                outputs[j]["Calendar_text"] = "[Calendar()"
                base_inputs = tokenizer(
                    outputs[j]["Calendar_text"] + "]" + "\n",
                    return_tensors="pt",
                )["input_ids"].cuda()
                outputs[j]["Calendar"] = self.calendar(calendar_string)
                outputs[j]["Calendar_output"] = [outputs[j]["Calendar_text"][1:], outputs[j]["Calendar"]]
                outputs[j]["Calendar_text"] = (
                    outputs[j]["Calendar_text"] + "->" + outputs[j]["Calendar"] + "]"
                )
                test_inputs = tokenizer(
                    outputs[j]["Calendar_text"] + "\n",
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
        for i in range((tokens.shape[1]-1)//N):
            if (N * (i + 1)) > tokens.shape[1]:
                continue
            input_tokens = tokens[:, (-N * (i + 1) - 1) : (-N * (i) - 1)]
            labels = tokens[
                :,
                int(tokens.shape[1] + (-N * (i + 1))) : int(tokens.shape[1] + (-N * i)),
            ]
            ret_tokens = tokens[:, : (-N * (i + 1) - 1)]
            print(tokens.shape)
            string = tokenizer.decode(input_tokens[0])
            # print(ret_strings)
            model_input = tokenizer(
                calendar_prompt.replace("<REPLACEGPT>", string) + string,
                return_tensors="pt",
            )["input_ids"]
            print(string)
            print(model_input.shape)
            with torch.no_grad():
                output = model(model_input.cuda()).logits.cpu()[:, -N:]
            new_outputs = self.generate_continuations(
                model_input,
                output,
                labels,
                model,
                tokenizer,
                dparser.parse(data["url"], fuzzy=True),
            )
            for output in new_outputs:
                if output is None:
                    continue
                output["index"] += int(tokens.shape[1] + (-N * (i + 1)))
                # filter by score
                if output["Score"] > 0.0:
                    outputs.append([output["Score"], output["index"]] + output["Calendar_output"])
        return outputs
