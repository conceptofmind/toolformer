import json

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    scores = dict()
    jsons = list()
    for i in range(8):
        with open(f"calc_data_{i}.json") as f:
            jsons.append(json.load(f))
    print(len(jsons))
    counter = 0
    for i in range(8):
        for item in jsons[i]:
            for output in item["calculator_outputs"]:
                if any(
                    [
                        "*" in output[2],
                        "/" in output[2],
                        "+" in output[2],
                        "-" in output[2],
                    ]
                ):
                    scores[output[0]] = scores.get(output[0], 0) + 1
                    counter += 1
    print(counter)
    sorted_keys = sorted(list(scores.keys()))
    running_values = [scores[sorted_keys[0]]]
    for i in range(1, len(sorted_keys)):
        running_values.append(running_values[-1] + scores[sorted_keys[0]])
    running_values = np.array(running_values)
    sorted_keys = np.array(sorted_keys)
    plt.plot(sorted_keys, (1.0 - (running_values / running_values[-1])) * counter)
    plt.xlabel("Score")
    plt.ylabel("Percentage of examples left")
    plt.show()
    # Thresholds:
    # 0.25 = 10% 0.58 = 1% 0.075 = 50%
    # call it 0.25 for now
