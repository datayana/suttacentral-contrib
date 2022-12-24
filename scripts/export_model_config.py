import os
import sys
import argparse
import logging

from transformers import AutoModelForMaskedLM


def main():
    """Parse args and run script"""
    # create parser for cli args
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of MLM HuggingFace model (ex: roberta-base)",
    )
    parser.add_argument(
        "--save_config",
        type=str,
        required=False,
        default=None,
        help="Path to a directory to export config.json",
    )

    # parse args
    args = parser.parse_args()

    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    if args.save_config:
        model.config.save_pretrained(args.save_config)


if __name__ == "__main__":
    main()
