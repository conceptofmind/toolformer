import json

if __name__ == "__main__":
    combined_data = dict()
    for i in range(8):
        with open(f"calc_data_{i}.json") as f:
            data = json.load(f)
        for item in data:
            for output in item["calculator_outputs"]:
                if any(
                    [
                        "*" in output[2],
                        "/" in output[2],
                        "+" in output[2],
                        "-" in output[2],
                    ]
                ):
                    if output[0] > 0.07:
                        if item["file_index"] not in list(combined_data.keys()):
                            combined_data[item["file_index"]] = dict()
                            combined_data[item["file_index"]]["text"] = item["text"]
                            combined_data[item["file_index"]]["outputs"] = list()
                        combined_data[item["file_index"]]["outputs"].append(
                            [output[1], output[2], output[3]]
                        )
        with open(f"calendar_data_{i}.json") as f:
            data = json.load(f)
        for item in data:
            for output in item["calendar_outputs"]:
                if output[0] > 0.25:
                    if item["file_index"] not in list(combined_data.keys()):
                        combined_data[item["file_index"]] = dict()
                        combined_data[item["file_index"]]["text"] = item["text"]
                        combined_data[item["file_index"]]["outputs"] = list()
                    combined_data[item["file_index"]]["outputs"].append(
                        [output[1], output[2], output[3]]
                    )
        with open(f"retrieval_data_{i}.json") as f:
            data = json.load(f)
        for item in data:
            for output in item["retrieval_outputs"]:
                if item["file_index"] not in list(combined_data.keys()):
                    combined_data[item["file_index"]] = dict()
                    combined_data[item["file_index"]]["text"] = item["text"]
                    combined_data[item["file_index"]]["outputs"] = list()
                combined_data[item["file_index"]]["outputs"].append(
                    [output[1], output[2], output[3]]
                )
    with open("../combined_data.json", "w") as f:
        json.dump(combined_data, f, indent=2)
