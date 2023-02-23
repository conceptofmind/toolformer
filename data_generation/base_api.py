import json
from typing import List
import torch
from transformers import (
    PreTrainedTokenizerBase,
    pipeline,
    PreTrainedModel,
    TextGenerationPipeline,
)
from torch import nn

MAX_BATCH_SIZE = 1  # My 3090 is weak 😔
N = 64  # SEQ Len
M = 16  # Min Loss Span To Consider


class APICallPostprocessing:
    def __init__(
        self,
        start_tokens: List[int],
        end_tokens: List[int],
        minimum_percentage: float = 0.1,
    ):
        """
        Base API Postprocesing class

        :param start_tokens: token representation for [ or other tokens
        :param end_tokens:  token representation for ] or other tokens
        :param minimum_percentage: pass percentage for candidate generation, less than this are ignored.
        """
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens
        self.minimum_percentage = minimum_percentage
        self.api_text = ""  # API text, might be better to pass it in
        self.k_values = 5  # Default topk generation, might be better to pass it in

    def filter_continuations(
        self,
        input_tokens: torch.Tensor,
        input_logits: torch.Tensor,
        labels: torch.Tensor,
        input_start: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Grab continuations that are valid

        :param input_tokens: tokenized inputs
        :param input_logits: input logits
        :param labels: labels for input logits
        :param input_start: start of real input
        :param tokenizer:
        :return: Values and Indices
        """
        # First, figure out locations...
        probs = torch.softmax(input_logits, dim=-1)
        # Make sure we don't keep any tokens that are supposed to be [
        remove_tokens = 1.0 - torch.sum(
            torch.stack([labels == start_token for start_token in self.start_tokens]),
            dim=0,
        )
        # Get maximum probability... Should be sufficient. Maybe switch to sum if there's issues later
        max_start_tokens = torch.amax(
            torch.stack(
                [probs[:, :, start_token] for start_token in self.start_tokens]
            ),
            dim=0,
        )
        max_start_tokens = max_start_tokens * remove_tokens
        return torch.topk(max_start_tokens[:, : -(M + 1)], k=self.k_values, dim=1)

    def create_candidates(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        input_tokens: torch.Tensor,
        labels: torch.Tensor,
        input_start: int,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        generator: TextGenerationPipeline,
        criterion: nn.CrossEntropyLoss,
    ):
        """
        Generates continuations of valid API calls

        :param indices: index to start
        :param values: values for filtering
        :param input_tokens: tokenized input
        :param labels: labels for input
        :param input_start: real start for base loss calculation
        :param model:
        :param tokenizer:
        :param generator: pipeline for text generation
        :param criterion: Should just be CE loss
        :return:
        """
        # Setup lists...
        outputs = list()
        num_to_keeps = list()
        texts_to_test = list()
        max_index = 0
        for i, batch in enumerate(indices):
            for j, index in enumerate(batch):
                if values[i][j] < self.minimum_percentage:
                    continue
                # Get base output
                base_outputs = model(input_tokens[:, input_start:].cuda()).logits[
                    :, index : index + M
                ]
                # Find starting location...
                num_keep = int(input_tokens[:, input_start:].shape[1] - index)
                # Calculate loss without API
                base_loss = criterion(
                    base_outputs.view(-1, base_outputs.size(-1)),
                    labels[:, index : index + M].cuda().view(-1),
                )
                # For padding later
                max_index = max(max_index, index)
                # API Text
                texts_to_test.append(
                    tokenizer.decode(input_tokens[:, : input_start + index][i])
                    + f" [{self.api_text}"
                )
                # grab 5 generations
                outputs.append(
                    generator(
                        texts_to_test[-1], max_new_tokens=28, num_return_sequences=5
                    )
                )
                # Add additional items to generation outputs...
                for k in range(5):
                    outputs[-1][k]["index"] = int(index)
                    outputs[-1][k]["base_loss"] = float(base_loss.item())
                    outputs[-1][k]["base_outputs"] = base_outputs
                # So we know where to look
                num_to_keeps.append(num_keep)
        return outputs, num_to_keeps, texts_to_test, max_index

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
        **kwargs,
    ):
        """
        Add API calls here.

        :param candidate: which candidate is being parsed
        :param outputs: individual candidate outputs
        :param texts_to_test: text for candidates
        :param tokenizer:
        :param input_tokens:
        :param input_start:
        :param nums_to_keep: values kept after generation
        :param base_loss: base loss value for candidate
        :param args: args to pass to subclass
        :param kwargs: kwargs to pass to subclass
        :return:
        """
        raise NotImplementedError("Fill this in with your API code please!")

    def generate_continuations(
        self,
        input_tokens: torch.Tensor,
        input_logits: torch.Tensor,
        labels: torch.Tensor,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *args,
        **kwargs,
    ):
        """
        Generate continuations

        :param input_tokens: input to model
        :param input_logits: output from model
        :param labels: labels for logits
        :param model:
        :param tokenizer:
        :param args: args to pass to add_api_calls
        :param kwargs: kwargs to pass to add_api_calls
        :return: individual candidate outputs
        """
        # Setup token stuff...
        input_start = input_tokens.shape[1] - input_logits.shape[1]
        start_str = tokenizer.decode(input_tokens[:, :input_start][0])
        # Find top tokens...
        values, indices = self.filter_continuations(
            input_tokens, input_logits, labels, input_start, tokenizer
        )
        # setup generation calls...
        generator = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, device=0
        )  # type: TextGenerationPipeline
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            outputs, num_to_keeps, texts_to_test, max_index = self.create_candidates(
                indices,
                values,
                input_tokens,
                labels,
                input_start,
                model,
                tokenizer,
                generator,
                criterion,
            )
            for i in range(len(outputs)):
                generated_texts, max_token_len, max_token_len_base = self.add_api_calls(
                    i,
                    outputs[i],
                    texts_to_test,
                    tokenizer,
                    input_tokens,
                    input_start,
                    num_to_keeps,
                    outputs[i][0]["base_loss"],
                    *args,
                    **kwargs,
                )
                if len(generated_texts) == 0:
                    outputs[i] = None
                    continue
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
                    generated_texts[j].append(
                        max_token_len_base - generated_texts[j][1].shape[1]
                    )
                    if generated_texts[j][-1] != 0:
                        generated_texts[j][1] = torch.cat(
                            (
                                generated_texts[j][1],
                                torch.zeros(
                                    (1, generated_texts[j][-1]),
                                    dtype=generated_texts[j][1].dtype,
                                    device=generated_texts[j][1].device,
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
                base_outputs = model(
                    torch.cat(
                        list(generated_text[1] for generated_text in generated_texts),
                        dim=0,
                    )
                ).logits
                best_loss = -99.0
                best_output = outputs[i][0]
                for j in range(len(generated_texts)):
                    num_to_keep = generated_texts[j][2]
                    if generated_texts[j][-2] != 0:
                        test = test_outputs[j][: -generated_texts[j][-2]]
                        test_loss = criterion(
                            test[-num_to_keep : -(num_to_keep - M)].view(
                                -1, generated_texts[j][-3]["base_outputs"].size(-1)
                            ),
                            labels[:, -num_to_keep : -(num_to_keep - M)]
                            .cuda()
                            .view(-1),
                        )
                    else:
                        test_loss = criterion(
                            test_outputs[j][-num_to_keep : -(num_to_keep - M)].view(
                                -1, generated_texts[j][-3]["base_outputs"].size(-1)
                            ),
                            labels[:, -num_to_keep : -(num_to_keep - M)]
                            .cuda()
                            .view(-1),
                        )
                    if generated_texts[j][-1] != 0:
                        base = base_outputs[j][: -generated_texts[j][-1]]
                        base_loss = criterion(
                            base[-num_to_keep : -(num_to_keep - M)].view(
                                -1, generated_texts[j][-3]["base_outputs"].size(-1)
                            ),
                            labels[:, -num_to_keep : -(num_to_keep - M)]
                            .cuda()
                            .view(-1),
                        )
                    else:
                        base_loss = criterion(
                            base_outputs[j][-num_to_keep : -(num_to_keep - M)].view(
                                -1, generated_texts[j][-3]["base_outputs"].size(-1)
                            ),
                            labels[:, -num_to_keep : -(num_to_keep - M)]
                            .cuda()
                            .view(-1),
                        )
                    generated_texts[j][-3]["generated_text"] = generated_texts[j][-3][
                        "generated_text"
                    ].replace(start_str, "")
                    if (
                        min(base_loss.item(), generated_texts[j][-3]["base_loss"])
                        - test_loss
                        > best_loss
                    ):
                        best_output = generated_texts[j][-3]
                        best_loss = generated_texts[j][-3]["base_loss"] - test_loss
                if len(generated_texts) > 0:
                    outputs[i] = best_output
                    outputs[i]["Score"] = float(best_loss.item())
                    outputs[i]["base_api_loss"] = float(base_loss.item())
                    del outputs[i]["base_outputs"]
                else:
                    outputs[i] = None
        # print(json.dumps(outputs, indent=2))
        return outputs

    def parse_article(
        self, data: dict, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
    ):
        """
        Takes in data dict and parses it into API continuations
        :param data: data, assuming it's from load_dataset and has a text field
        :param model:
        :param tokenizer:
        :return: outputs for the input data, should have index of API call insertion, API, and score value at minimum.
        """
        raise NotImplementedError("Fill this in for what you need to do please!")
