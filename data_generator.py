import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset
from prompts import retrieval_prompt
from data_generation.retrieval import RetrievalPostprocessing
from data_generation.calendar import CalendarPostprocessing
from data_generation.calculator import CalculatorPostprocessing
from data_generation.api_checker import check_apis_available


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

