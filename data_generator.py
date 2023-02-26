import re
from dataclasses import dataclass

import torch
import torch.nn.functiional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from typing import Optional, Tuple, Dict, Callable, Iterator

from datasets import load_dataset
from prompts import retrieval_prompt
from data_generation.retrieval import RetrievalPostprocessing
from data_generation.calendar import CalendarPostprocessing
from data_generation.calculator import CalculatorPostprocessing
from data_generation.api_checker import check_apis_available

"""
Tool use parsing regex
<API>Calculator(1 / 2) -> 0.5</API> into ("Calculator", "400 / 1400", "0.5")
<API>Date()</API> into ("Date", None, None)
<API>WikiSearch('abcdef')</API> into ("WikiSearch", "'abcdef'", None)
"""
TOOL_REGEX = "<API>([A-Za-z]+)\((.*)\)(\s->\s.*)?</API>"

def interpret_tools(input_text: str, tools: Dict[str, Callable[[str], str]]) -> str:
    """
    Interprets the use of tools in text and replaces them with their output.
    """
    output_text = input_text
    for match in re.finditer(TOOL_REGEX, input_text):
        tool_name = match.group(1)
        tool_args = match.group(2) # empty string if no args
        tool_output: Optional[str] = match.group(3)
        if tool_name in tools and tool_output is None:
            args = [tool_args] if tool_args else []
            tool_output = tools[tool_name](*args)
            output = f"<API>{tool_name}({tool_args}) -> {tool_output}</API>"
            output_text = output_text.replace(match.group(0), output)
        elif tool_name not in tools:
            print(f"Unknown tool: {tool_name}")
    
    return output_text




'''
To sample API calls, we write a prompt that encourages a LM to annotate text with API calls.
Then, we find the top k locations with the highest probability of <API> tokens.
Then, we sample m API calls from the top k locations, giving the prompt <API> as a prefix and </API> as end of sentence token.

Then we execute all found API calls and keep their results.
Then we filter API calls by measuring the cross entropy loss between the original text with API call and results prefixed to it, 
the original text with no call, and the original text with the API call args but without outputs.
'''
def sample_api_calls(
    api_name: str,
    api: Callable[[str], str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    annotation_prompt: str,
    prompt: str,
    k: int, 
    m: int,
    start_token: int,
    end_token: int
):
    prompted_text = annotation_prompt.replace("<REPLACEGPT>", prompt) + prompt
    prompted_tokens = tokenizer(prompted_text, return_tensors="pt")["input_ids"].cuda()

    with torch.no_grad():
        logits = model(input_ids=prompted_tokens).logits
    
    # Find top k locations with highest probability of <API> tokens
    logits_for_api = logits[0, len(prompted_tokens):, start_token]
    top_k_logits, top_k_indices = torch.topk(logits_for_api, k)
    
    for idx in top_k_indices:
        continuation_start = idx + len(prompted_tokens)
    
        api_calls = model.generate(
            prompted_tokens[:, :continuation_start],
            generation_config=GenerationConfig(
                do_sample=True,
                max_length=100 + continuation_start,
                eos_token_id=end_token,
                num_return_sequences=m,
            ),
        ).sequences

        for api_call in api_calls:
            api_call_str = tokenizer.decode(api_call[continuation_start:])
            
            match = re.match(TOOL_REGEX, api_call_str)
            tool_name = match.group(1)
            tool_args = match.group(2)

            if tool_name == api_name:
                api_output_str = api(tool_args)

                yield tool_name, tool_args, api_output_str

def api_loss_reduction(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, tool_name, tool_args, tool_output, prompt, weights=None):
    with_api =  f"<API>{tool_name}({tool_args}) -> {tool_output}</API>{prompt}"
    with_args = f"<API>{tool_name}({tool_args})</API>{prompt}"
    without_api = f"{prompt}"
    prompt_tokens = tokenizer([with_api, with_args, without_api], return_tensors="pt")["input_ids"].cuda()
    with torch.no_grad():
        logits = model(input_ids=prompt_tokens).logits
        losses = F.cross_entropy(logits, prompt_tokens, reduction="none", weight=weights)
        L_plus = losses[0].sum()
        L_minus = min(losses[1].sum(), losses[2].sum())
        return L_minus - L_plus




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
    api_handler = CalculatorPostprocessing(start_tokens, end_tokens)
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).cuda()
    dataset = load_dataset("c4", "en", split="train", streaming=True)
    iter_data = iter(dataset)
    test = False
    counter = 0
    while counter < 10:
        data = next(iter_data)
        available = check_apis_available(data, gpt_tokenizer)
        test = available.calculator
        if test:
            api_handler.parse_article(data, model, gpt_tokenizer)
            counter += 1
