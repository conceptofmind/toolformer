from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase


@dataclass
class AvailableAPIs:
    """Keeps track of available APIs"""

    retrieval: bool = True

    def check_any_available(self):
        return any([self.retrieval])


def check_apis_available(
    data: dict, tokenizer: PreTrainedTokenizerBase
) -> AvailableAPIs:
    """
    Returns available APIs with boolean flags

    :param data: from load_dataset, assumes ['text'] is available
    :param tokenizer: Tokenizer to tokenize data
    :return: AvailableAPIs
    """
    tokenized_data = tokenizer(data["text"])["input_ids"]
    available = AvailableAPIs()
    if len(tokenized_data) < 8000:
        available.retrieval = False
    return available
