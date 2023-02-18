import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase, AutoTokenizer, PreTrainedModel, pipeline
from datasets import load_dataset
from dataclasses import dataclass
import nltk
from nltk import tokenize
from prompts import retrieval_prompt
from typing import List
nltk.download('punkt')

@dataclass
class AvailableAPIs:
    '''Keeps track of available APIs'''
    retrieval: bool = True

    def check_any_available(self):
        return any([self.retrieval])


class APICallPostprocessing:
    def __init__(self, start_tokens:List[int], end_tokens:List[int]):
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens

    def find_and_rank(self,
                      input_tokens: torch.Tensor,
                      input_logits: torch.Tensor,
                      labels: torch.Tensor,
                      api_text: str,
                      model: PreTrainedModel,
                      tokenizer:PreTrainedTokenizerBase,
                      ):
        # First, figure out locations...
        input_start = input_tokens.shape[1] - input_logits.shape[1]
        start_str = tokenizer.decode(input_tokens[:, :input_start][0])
        probs = torch.softmax(input_logits, dim=-1)
        remove_tokens = 1.0 - torch.sum(torch.stack([labels==start_token for start_token in self.start_tokens]), dim=0)
        # print(remove_tokens)
        max_start_tokens = torch.amax(torch.stack(
            [probs[:, :, start_token] for start_token in self.start_tokens]), dim=0)
        # remove tokens where it's appropriate to be the start token
        max_start_tokens = max_start_tokens * remove_tokens
        # Each sequence find top 5...
        values, indicies = torch.topk(max_start_tokens, k=5, dim=1)
        #for each...
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)
        # setup generation calls...
        for i, batch in enumerate(indicies):
            for index in batch:
                text = tokenizer.decode(input_tokens[:, :input_start+index][i]) + f" [{api_text}"
                print(text)
                outputs = generator(text, max_length=input_start+index+28, num_return_sequences=5)
                for j in range(len(outputs)):
                    outputs[j]['generated_text'] = outputs[j]['generated_text'].replace(start_str, "")
                print(outputs)




def check_apis_available(data: dict, tokenizer: PreTrainedTokenizerBase) -> AvailableAPIs:
    tokenized_data = tokenizer(data['text'])['input_ids']
    # print(len(tokenized_data))
    available = AvailableAPIs()
    if len(tokenized_data) < 8000:
        available.retrieval = False
    return available



if __name__ == '__main__':
    gpt_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    prompt_tokens = gpt_tokenizer(retrieval_prompt, return_tensors='pt')['input_ids']
    start_tokens = [gpt_tokenizer("[")['input_ids'][0], gpt_tokenizer(" [")['input_ids'][0]]
    end_tokens = [gpt_tokenizer("]")['input_ids'][0], gpt_tokenizer(" ]")['input_ids'][0]]  # TODO: keep second?
    api_handler = APICallPostprocessing(start_tokens, end_tokens)
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
    dataset = load_dataset("c4", "en", split='train', streaming=True)
    iter_data = iter(dataset)
    test = False
    while not test:
        data = next(iter_data)
        available = check_apis_available(data, gpt_tokenizer)
        test = available.retrieval
    # print(data)
    tokens = gpt_tokenizer(data['text'], return_tensors='pt')['input_ids']
    input_tokens = tokens[:, -301:-1]
    labels = tokens[:, -300:]
    ret_tokens = tokens[:, :-301]
    print(tokens.shape)
    string = gpt_tokenizer.decode(input_tokens[0])
    ret_strings = tokenize.sent_tokenize(gpt_tokenizer.decode(ret_tokens[0]))
    model_input = gpt_tokenizer(retrieval_prompt.replace("<REPLACEGPT>", string) + string, return_tensors='pt')['input_ids']
    print(string)
    print(model_input.shape)
    with torch.no_grad():
        output = model(model_input.cuda()).logits.cpu()[:, -300:]
    api_handler.find_and_rank(model_input, output, labels, "Retrieval(", model, gpt_tokenizer)
