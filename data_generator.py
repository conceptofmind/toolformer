import re
import random
from dataclasses import dataclass


import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from typing import Optional, Tuple, Dict, Callable, Iterator

from datasets import load_dataset
from prompts import calculator_prompt, wikipedia_search_prompt
from tools import Tool, Calculator, WikiSearch
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
        tool_args = match.group(2) # empty string if no args
        tool_output: Optional[str] = match.group(3)
        if tool_name in tools and tool_output is None:
            args = [tool_args] if tool_args else []
            tool_output = tools[tool_name](*args)
            output = f"[{tool_name}({tool_args}) -> {tool_output}]"
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
    tool: Tool,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    k: int, 
    m: int,
    start_tokens: int,
    end_token: int,
    api_call_threshold: float = 0.1
):
    
    prompted_text = tool.prompt.replace("<REPLACEGPT>", prompt) + prompt
    prompted_tokens = tokenizer(prompted_text, return_tensors="pt", truncation=True).to(model.device)
    prefix_tokens = tokenizer(tool.prompt.replace("<REPLACEGPT>", prompt), return_tensors="pt", truncation=True)['input_ids'].to(model.device)
    with torch.no_grad():
        logits = model(**prompted_tokens).logits
        probs = F.softmax(logits, dim=-1)
    
    # Find top k locations with highest probability of <API> tokens
    probs_for_api = probs[0, prefix_tokens.shape[1]:, start_tokens].max(dim=-1).values
    top_k_probs, top_k_indices = torch.topk(probs_for_api, min(len(probs_for_api), k))
    
    for idx, prob in zip(top_k_indices, top_k_probs):
        if prob.item() < api_call_threshold:
            break

        continuation_start = idx + prefix_tokens.shape[1]
    
        api_calls = model.generate(
            prompted_tokens['input_ids'][:, :continuation_start],
            generation_config=GenerationConfig(
                do_sample=True,
                max_new_tokens=100,
                eos_token_id=end_token,
                num_return_sequences=m,
            ),
        )

        for api_call in api_calls:
            api_call_str = tokenizer.decode(api_call[continuation_start:])
            
            match = re.match(TOOL_REGEX, api_call_str)
            if match is None:
                continue
            tool_name = match.group(1)
            tool_args = match.group(2)

            if tool_name == tool.__class__.__name__:
                api_output_str = tool(tool_args)

                yield tool_name, tool_args, api_output_str

def api_loss_reduction(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, tool_name, tool_args, tool_output, prompt, weights=None):
    with_api =  f"[{tool_name}({tool_args}) -> {tool_output}]{prompt}"
    with_args = f"[{tool_name}({tool_args})]{prompt}"
    without_api = f"{prompt}"
    prompt_tokens = tokenizer([with_api, with_args, without_api], return_tensors="pt")["input_ids"].cuda()
    with torch.no_grad():
        logits = model(input_ids=prompt_tokens).logits
        losses = F.cross_entropy(logits, prompt_tokens, reduction="none", weight=weights)
        L_plus = losses[0].sum()
        L_minus = min(losses[1].sum(), losses[2].sum())
        return L_minus - L_plus



if __name__ == "__main__":
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    gpt_tokenizer.truncation_side = "left"
    start_tokens = [
        gpt_tokenizer("[")["input_ids"][0],
        gpt_tokenizer(" [")["input_ids"][0],
    ]
    end_token = gpt_tokenizer("]")["input_ids"][0]

    tools = [Calculator(), WikiSearch()]

    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
    ).cuda()

    dataset = load_dataset("c4", "en", split="train", streaming=True)
    iter_data = iter(dataset)
    test = False
    counter = 0
    while counter < 10:
        data = next(iter_data)
        for tool in tools:
            if tool.heuristic(data):
                for tool_name, tool_args, tool_output in sample_api_calls(
                    # assumes tool's name is the same as the name of the class implementing it
                    tool,
                    model,
                    gpt_tokenizer,
                    data["text"],
                    5,
                    5,
                    start_tokens,
                    end_token,
                    0.1,
                ):
                    print(tool_name, tool_args, tool_output)
                    print(api_loss_reduction(model, gpt_tokenizer, tool_name, tool_args, tool_output, data["text"]))
