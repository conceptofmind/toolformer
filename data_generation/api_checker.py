from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
import dateutil.parser as dparser
import random
import re


@dataclass
class AvailableAPIs:
    """Keeps track of available APIs"""

    retrieval: bool = True
    calendar: bool = True
    calculator: bool = True
    llmchain: bool = True

    def check_any_available(self):
        return any([self.retrieval, self.calendar, self.calculator])


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
    # In case we need a different version, found this here:
    # https://stackoverflow.com/questions/28198370/regex-for-validating-correct-input-for-calculator
    calc_pattern = re.compile("^(\d+[\+\-\*\/]{1})+\d+$")
    if len(tokenized_data) < 4096:
        available.retrieval = False
    try:
        date = dparser.parse(data["url"], fuzzy=True)
    except (ValueError, OverflowError):
        available.calendar = False
    available.calculator = False
    tried_rand = False
    for i in range(len(tokenized_data) // 100):
        text = tokenizer.decode(tokenized_data[i * 100 : (i + 1) * 100])

        operators = bool(re.search(calc_pattern, text))
        equals = any(
            ["=" in text, "equal to" in text, "total of" in text, "average of" in text]
        )
        if not (operators and equals) and not tried_rand:
            tried_rand = True
            text = text.replace("\n", " ")
            text = text.split(" ")
            text = [item for item in text if item.replace(".", "", 1).isnumeric()]
            if len(text) >= 3:
                if random.randint(0, 99) == 0:
                    available.calculator = True
        else:
            available.calculator = True

    return available
