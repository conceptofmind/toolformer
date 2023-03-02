import copy
import requests
import calendar
import json
import torch
import wolframalpha
import openai
import datetime
import time
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModel,
    T5ForConditionalGeneration,
)
from typing import List
from operator import truediv, mul, add, sub
from langchain.chains import LLMChain
from langchain import Cohere, PromptTemplate

# Optional imports
from googleapiclient.discovery import build


"""
Calendar

Uses Python's datetime and calendar libraries to retrieve the current date.

input - None

output - A string, the current date.
"""


def Calendar(date=datetime.datetime.now()):
    return f"Today is {calendar.day_name[date.weekday()]}, {calendar.month_name[date.month]} {date.day}, {date.year}."


"""
retrieval

Uses Carptriever to retrieve sentences before the current context.

input_sentences - List[String], sentences to retrieve from
input_text - String, the input text (e.g. The dog's name is)
k - The number of sentences to retrieve

output - A list of strings, each string is the retrieved sentence, and the sentence after.
"""


class Retriever:
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            "CarperAI/carptriever-1", add_pooling_layer=False
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained("CarperAI/carptriever-1")

    def retrieval(
        self, input_sentences: List[str], input_text: str, k: int
    ) -> List[str]:
        if k > len(input_sentences):
            # I'd error but LMs do stupid stuff sometimes
            return input_sentences
        input_sentences = copy.deepcopy(input_sentences)
        input_sentences.append(input_text)
        output_list = []
        for sentence in input_sentences:
            inputs = self.tokenizer(
                sentence, padding=True, truncation=True, return_tensors="pt"
            )
            # print(inputs)
            inputs["input_ids"] = inputs["input_ids"].cuda()
            inputs["token_type_ids"] = inputs["token_type_ids"].cuda()
            inputs["attention_mask"] = inputs["attention_mask"].cuda()
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = mean_pooling(outputs[0], inputs["attention_mask"])
            output_list.append(embeddings)
        query_embedding, sentence_embeddings = output_list[-1], torch.concat(
            output_list[:-1], 0
        )
        # print(len(sentence_embeddings), sentence_embeddings[0].shape)
        scores = (query_embedding @ sentence_embeddings.transpose(0, 1)).cpu().tolist()
        # print(scores)
        sentence_score_pairs = sorted(
            zip(input_sentences[:-1], scores[0]), reverse=True, key=lambda x: x[1]
        )
        continued_sentence_score_pairs = sorted(
            zip(input_sentences[1:], scores[0]), reverse=True, key=lambda x: x[1]
        )
        # print(sentence_score_pairs)
        return [
            sentence_pair[0] + " " + continue_pair[0]
            for sentence_pair, continue_pair in zip(
                sentence_score_pairs[:k], continued_sentence_score_pairs[:k]
            )
        ]


def mean_pooling(token_embeddings: torch.Tensor, mask: torch.Tensor):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


"""
Wikipedia Search

Uses ColBERTv2 to retrieve Wikipedia documents.

input_query - A string, the input query (e.g. "what is a dog?")
k - The number of documents to retrieve

output - A list of strings, each string is a Wikipedia document

Adapted from Stanford's DSP: https://github.com/stanfordnlp/dsp/
Also see: https://github.com/lucabeetz/dsp
"""


class ColBERTv2:
    def __init__(self, url: str):
        self.url = url

    def __call__(self, query, k=1):
        topk = colbertv2_get_request(self.url, query, k)

        topk = [doc["text"] for doc in topk]
        return topk


def colbertv2_get_request(url: str, query: str, k: int):
    payload = {"query": query, "k": k}
    res = requests.get(url, params=payload)

    topk = res.json()["topk"][:k]
    return topk


def WikiSearch(input_query: str):
    k = 10
    retrieval_model = ColBERTv2(
        "http://ec2-44-228-128-229.us-west-2.compute.amazonaws.com:8893/api/search"
    )
    output = retrieval_model(input_query, k)
    return output


"""
Machine Translation - NLLB-600M

Uses HuggingFace's transformers library to translate input query to English.

input_query - A string, the input query (e.g. "what is a dog?")

output - A string, the translated input query.
"""


def MT(input_query: str):
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    input_ids = tokenizer(input_query, return_tensors="pt")
    outputs = model.generate(
        **input_ids,
        forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"],
    )
    output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return output


"""
Calculator

Calculates the result of a mathematical expression.

input_query - A string, the input query (e.g. "400/1400")

output - A float, the result of the calculation

Adapted from: https://levelup.gitconnected.com/3-ways-to-write-a-calculator-in-python-61642f2e4a9a 
"""


def Calculator(input_query: str):
    operators = {"+": add, "-": sub, "*": mul, "/": truediv}
    if input_query.isdigit():
        return float(input_query)
    for c in operators.keys():
        left, operator, right = input_query.partition(c)
        if operator in operators:
            return round(operators[operator](Calculator(left), Calculator(right)), 2)


# Other Optional Tools

"""
LangChain LLMChain

input_question - A string, the input query (e.g. "what is a dog?")

output - String for generation

Requires that you set your COHERE_API_KEY environment variable before starting.
"""
def langchain_llmchain(input_question):
    # TODO: Check succinct if it's good once we don't have rate limited APIs
    template = """Please be succinct in your answer to this question.
Question: {question}

Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = Cohere(model="command-xlarge-nightly")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.predict(question=input_question)


"""
HuggingFace API

Uses HuggingFace's API to generate text.

input_query - A string, the input query (e.g. "what is a dog?")

output - A string, the generated text

API_TOKEN - your HuggingFace API token
"""


def HuggingfaceAPI(input_query: str):
    model_id = "gpt-neox-20b"
    API_TOKEN = "YOUR_API_TOKEN"
    API_URL = "https://api-inference.huggingface.co/models/{model_id}".format(
        model_id=model_id
    )
    headers = {"Authorization": f"Bearer {API_TOKEN}".format(API_TOKEN=API_TOKEN)}

    def query(payload):
        data = json.dumps(payload)
        response = requests.request("POST", API_URL, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))

    data = query(input_query)
    return data[0]["generated_text"]


"""
Wolfram Alpha Calculator

pip install wolframalpha

Uses Wolfram Alpha API to calculate input query.

input_query - A string, the input query (e.g. "what is 2 + 2?")

output - A string, the answer to the input query

wolfarm_alpha_appid - your Wolfram Alpha API key
"""


def WolframAlphaCalculator(input_query: str):
    wolfram_alpha_appid = "YOUR_WOLFRAM_ALPHA_APPID"
    wolfram_client = wolframalpha.Client(wolfram_alpha_appid)
    res = wolfram_client.query(input_query)
    assumption = next(res.pods).text
    answer = next(res.results).text
    return f"Assumption: {assumption} \nAnswer: {answer}"


"""
Google Search

Uses Google's Custom Search API to retrieve Google Search results.

input_query - The query to search for.
num_results - The number of results to return.
api_key - Your Google API key.
cse_id - Your Google Custom Search Engine ID.

output - A list of dictionaries, each dictionary is a Google Search result
"""


def custom_search(query, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, **kwargs).execute()
    return res["items"]


def google_search(input_query: str):
    api_key = "YOUR_GOOGLE_API_KEY"
    cse_id = "YOUR_GOOGLE_CSE_ID"
    num_results = 10
    metadata_results = []
    results = custom_search(
        input_query, num=num_results, api_key=api_key, cse_id=cse_id
    )
    for result in results:
        metadata_result = {
            "snippet": result["snippet"],
            "title": result["title"],
            "link": result["link"],
        }
        metadata_results.append(metadata_result)
    return metadata_results


"""
SteamSHP

Uses HuggingFace's transformers library to generate text.

input_query - A string, the input query (e.g. "what is a dog?")

output - A list of strings, the generated text

"""


def SteamSHP(input_query: str):
    device = "cuda"  # if you have a GPU
    tokenizer = AutoTokenizer.from_pretrained("stanfordnlp/SteamSHP-flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained(
        "stanfordnlp/SteamSHP-flan-t5-large"
    ).to(device)
    x = tokenizer([input_query], return_tensors="pt").input_ids.to(device)
    y = model.generate(x, max_new_tokens=1)
    output = tokenizer.batch_decode(y, skip_special_tokens=True)
    return output


"""
Goose AI

pip install openai

Uses GPT-NeoX 20B to generate text.

input_query - A string, the input query (e.g. "what is a dog?")

output - A string, the generated text

openai.api_key - your GooseAI API key
"""


def GooseAI(input_query: str):
    openai.api_key = "YOUR_API_KEY"
    openai.api_base = "https://api.goose.ai/v1"
    # Create a completion, return results streaming as they are generated.
    # Run with `python3 -u` to ensure unbuffered output.
    completion = openai.Completion.create(
        engine="gpt-neo-20b", prompt=input_query, max_tokens=160
    )
    return completion.choices[0].text


"""
Bing Search

Uses Bing's Custom Search API to retrieve Bing Search results.

input_query: The query to search for.
bing_subscription_key: Your Bing API key.
num_results: The number of results to return.

output: A list of dictionaries, each dictionary is a Bing Search result
"""


def _bing_search_results(search_term: str, bing_subscription_key: str, count: int):
    headers = {"Ocp-Apim-Subscription-Key": bing_subscription_key}
    params = {
        "q": search_term,
        "count": count,
        "textDecorations": True,
        "textFormat": "HTML",
    }
    response = requests.get(
        "https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params
    )
    response.raise_for_status()
    search_results = response.json()
    return search_results["webPages"]["value"]


def bing_search(input_query: str):
    bing_subscription_key = "YOUR BING API KEY"
    num_results = 10
    metadata_results = []
    results = _bing_search_results(
        input_query, bing_subscription_key, count=num_results
    )
    for result in results:
        metadata_result = {
            "snippet": result["snippet"],
            "title": result["name"],
            "link": result["url"],
        }
        metadata_results.append(metadata_result)
    return metadata_results


if __name__ == "__main__":
    print(langchain_llmchain("Please respond"))

    print(
        WikiSearch("What is a dog?")
    )  # Outputs a list of strings, each string is a Wikipedia document

    print(Calendar())  # Outputs a string, the current date

    print(Calculator("400/1400"))  # For Optional Basic Calculator

    print(MT("Un chien c'est quoi?"))  # What is a dog?

    # Optional Tools

    print(
        HuggingfaceAPI("What is a dog?")
    )  # Outputs a string, the answer to the input query

    print(SteamSHP("What is a dog?"))  # Outputs a list with an answer

    print(WolframAlphaCalculator("What is 2 + 2?"))  # 4

    print(GooseAI("What is a dog?"))  # Outputs a string, the answer to the input query

    print(google_search("What is a dog?"))
    # Outputs a list of dictionaries, each dictionary is a Google Search result

    print(bing_search("What is a dog?"))
    # Outputs a list of dictionaries, each dictionary is a Bing Search result
