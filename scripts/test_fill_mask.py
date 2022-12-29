import sys
import argparse
from transformers import AutoTokenizer
from transformers import pipeline


def main():
    """Parse args and run script"""
    # create parser for cli args
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model dir (config.json, checkpoints, etc).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=False,
        default=None,
        help="Path to the tokenizer config (default: use --model).",
    )

    # parse args
    args = parser.parse_args()

    # create a fill mask pipeline
    fill_mask = pipeline(
        "fill-mask", model=args.model, tokenizer=args.tokenizer or args.model
    )

    # load separately to show off
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

    test_cases = [
        "Evaṁ me [MASK]—",
        "Evaṁ [MASK] sutaṁ—",
        "[MASK] me sutaṁ—",
        "Yaṁ kiñci samudayadhammaṁ sabbaṁ taṁ [MASK].",
    ]

    for case in test_cases:
        print(f"*** {case}")
        print("tokenizer output> " + str(tokenizer(case)))
        predictions = fill_mask(case)
        for result in predictions:
            print("pred> " + str(result))


if __name__ == "__main__":
    main()
