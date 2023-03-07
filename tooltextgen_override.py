


import torch
from typing import Optional, List, Union, Dict, Any
import warnings
from torch import nn
import types
from transformers import PreTrainedModel, LogitsProcessorList, StoppingCriteriaList, AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.generation.utils import SampleOutput, dist, validate_stopping_criteria, SampleEncoderDecoderOutput, SampleDecoderOnlyOutput, ModelOutput


def prepare_inputs_wrapper(tool_names, tools, model: PreTrainedModel):
    def sample_fix_toolformer(
            self,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            logits_warper: Optional[LogitsProcessorList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: Optional[bool] = False,
            **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.sample`] directly. Use generate() instead.
        For an overview of generation strategies and code examples, check the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.SampleDecoderOnlyOutput`], [`~generation.SampleEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.SampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.SampleEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     TopKLogitsWarper,
        ...     TemperatureLogitsWarper,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.config.pad_token_id = model.config.eos_token_id
        >>> model.generation_config.pad_token_id = model.config.eos_token_id

        >>> input_prompt = "Today is a beautiful day, and"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> # instantiate logits processors
        >>> logits_warper = LogitsProcessorList(
        ...     [
        ...         TopKLogitsWarper(50),
        ...         TemperatureLogitsWarper(0.7),
        ...     ]
        ... )

        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
        >>> outputs = model.sample(
        ...     input_ids,
        ...     logits_processor=logits_processor,
        ...     logits_warper=logits_warper,
        ...     stopping_criteria=stopping_criteria,
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Today is a beautiful day, and a wonderful day.\n\nI was lucky enough to meet the']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs, extra_num = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # TODO: Figure out a way to make this automatic
            # Fix for shoving in additional tokens, since it could go over length
            if (input_ids.shape[1]) + (model_inputs["input_ids"].shape[1] - 1) > 2048:
                if return_dict_in_generate:
                    if self.config.is_encoder_decoder:
                        return SampleEncoderDecoderOutput(
                            sequences=input_ids,
                            scores=scores,
                            encoder_attentions=encoder_attentions,
                            encoder_hidden_states=encoder_hidden_states,
                            decoder_attentions=decoder_attentions,
                            cross_attentions=cross_attentions,
                            decoder_hidden_states=decoder_hidden_states,
                        )
                    else:
                        return SampleDecoderOnlyOutput(
                            sequences=input_ids,
                            scores=scores,
                            attentions=decoder_attentions,
                            hidden_states=decoder_hidden_states,
                        )
                else:
                    return input_ids
            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            if (extra_num):
                input_ids = torch.cat([input_ids, model_inputs["input_ids"][:,-extra_num:], next_tokens[:, None]], dim=-1)
            else:
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, input_ids, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

    def process_tools(input_ids):
        tokenizer = AutoTokenizer.from_pretrained("customToolformer")
        # TODO: Support more than 1 batch size
        tool_string = tokenizer.decode(input_ids[0,:-1])
        call_name = tool_string.split("<TOOLFORMER_API_START>")[-1].split("(")[0]
        data = tool_string.split("<TOOLFORMER_API_START>")[-1].split("(")[1].replace("\"", "").split(")")[0]
        extra_tokens = None
        for i, tool_name in enumerate(tool_names):
            if call_name == tool_name:
                extra_tokens = tokenizer(tools[i](data) + "<TOOLFORMER_API_END>", return_tensors="pt")["input_ids"]
        if extra_tokens is None:
            return tokenizer("<TOOLFORMER_API_END>", return_tensors="pt")["input_ids"]
        else:
            return extra_tokens


    def prepare_inputs_for_generation_toolformer(self, input_ids, past_key_values=None, **kwargs):
        if input_ids.shape[0] > 1:
            raise NotImplementedError(f"Only supporting batch size 1 for now! Sorry! You tried to send in {input_ids.shape}")
        extra_tokens = None
        if input_ids[0, -1] == 50258:
            extra_tokens = process_tools(input_ids)
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        extra_num = None
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if extra_tokens is not None:
                extra_num = extra_tokens.shape[1]
                input_ids = torch.cat((input_ids, extra_tokens.to(input_ids.device)), dim=-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            if extra_tokens is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((1, extra_tokens.shape[1]))], dim=-1
                )
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                if extra_tokens is not None:
                    position_ids = position_ids[:, -(1 + extra_tokens.shape[1]):]
                else:
                    position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }, extra_num

    def _update_model_kwargs_for_generation_toolformer(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        input_ids: Any,
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)
        # print(outputs.logits.shape[1])
        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                if model_kwargs["past_key_values"] is not None:
                    model_kwargs["attention_mask"] = torch.cat(
                        [attention_mask,
                         attention_mask.new_ones((attention_mask.shape[0], input_ids.shape[1]-attention_mask.shape[1]))
                         ], dim=-1
                    )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], input_ids.shape[1]-attention_mask.shape[1]))],
                    dim=-1,
                )

        return model_kwargs
    model.sample = types.MethodType(sample_fix_toolformer, model)
    model.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation_toolformer, model)
    model._update_model_kwargs_for_generation = types.MethodType(_update_model_kwargs_for_generation_toolformer, model)


if __name__ == '__main__':
    text_example = """User: What is the weather today?
Assistant: <TOOLFORMER_API_START>Location()<TOOLFORMER_API_RESPONSE>Chicago, Illinois<TOOLFORMER_API_END><TOOLFORMER_API_START>Weather("Chicago, Illinois")<TOOLFORMER_API_RESPONSE>60 degrees, windy<TOOLFORMER_API_END> It is currently 60 degrees and windy.
User: What is the weather in Minneapolis, Minnesota?
Assistant: <TOOLFORMER_API_START>Weather("Minneapolis, Minnesota")<TOOLFORMER_API_RESPONSE>45 degrees, raining<TOOLFORMER_API_END> It is currently 45 degrees and raining.
User: What is the weather in at the Golden Gate Bridge?
Assistant: <TOOLFORMER_API_START>Location("Golden Gate Bridge")<TOOLFORMER_API_RESPONSE>Chicago, Illinois<TOOLFORMER_API_END><TOOLFORMER_API_START>Weather("Chicago, Illinois")<TOOLFORMER_API_RESPONSE>67 degrees, sunny<TOOLFORMER_API_END> It is currently 67 degrees and sunny.
User: Where is the Louvre?
Assistant:"""
    tokenizer = AutoTokenizer.from_pretrained(r"dmayhem93/toolformer_v0_epoch3")
    model = AutoModelForCausalLM.from_pretrained(
        r"dmayhem93/toolformer_v0_epoch3",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).cuda()

    def location_tool(x):
        return "Paris, France"

    prepare_inputs_wrapper(["Location"], [location_tool], model)
    generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=0
    )
    for i in range(5):
        items = generator(
            text_example, max_new_tokens=32, num_return_sequences=1,
        )
        for item in items:
            print(item["generated_text"])
            print("---")