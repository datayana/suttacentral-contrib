import torch
from transformers import AutoTokenizer
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./models/palibert/v0.1",
    tokenizer="./models/palibert/v0.1"
)

tokenizer = AutoTokenizer.from_pretrained("./models/palibert/v0.1")

test_cases = [
    "Evaṁ me [MASK]—",
    "Evaṁ [MASK] sutaṁ—",
    "[MASK] me sutaṁ—",
    "Yaṁ kiñci samudayadhammaṁ sabbaṁ taṁ [MASK]."
]

for case in test_cases:
    print("*** {case}")
    print("tokenizer output> "+str(tokenizer(case)))
    predictions = fill_mask(case)
    for result in predictions:
        print("pred> "+str(result))
