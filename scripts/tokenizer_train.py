"""Train a BPE tokenizer on a Pali text from Sutta Central using HuggingFace."""
import os
import sys
import argparse
import logging

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import (
    CharDelimiterSplit,
    Punctuation,
    Whitespace,
    Sequence,
)
from tokenizers import AddedToken
from tokenizers.normalizers import Lowercase
from tokenizers.processors import BertProcessing
from transformers import PreTrainedTokenizerFast
from tokenizers.processors import TemplateProcessing

from suttacentral.contrib.data.samples import PALI_CATCHPHRASES


def main():
    """Parse args and run script"""
    # create parser for cli args
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to the file to train BPE.",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        required=False,
        default=None,
        help="Path to a file where to export the tokeniser in json format.",
    )
    parser.add_argument(
        "--save_as_pretrained",
        type=str,
        required=False,
        default=None,
        help="Path to a file to save the tokenizer as Fast object.",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        required=False,
        default=None,
        help="Path to a directory to save the tokenizer model (vocab.json, merges.txt).",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="print DEBUG logs",
    )

    # parse args
    args = parser.parse_args()

    # set up logger level
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    handler.setFormatter(log_format)
    logger.addHandler(handler)

    # Initialize
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Sequence(
        [
            Whitespace(),
            CharDelimiterSplit("\n"),
            Punctuation(),
        ]
    )
    tokenizer.normalizer = Lowercase()

    # Create trainer
    trainer = BpeTrainer(
        vocab_size=30000,
        min_frequence=2,
        show_progress=True,
        special_tokens=["<pad>", "<unk>", "[CLS]", "[SEP]", AddedToken("[MASK]", lstrip=True, normalized=False)],
    )

    # Train the tokenizer
    tokenizer.train([args.train_file], trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    # convert to transformers tokenizer
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=512,
        mask_token=AddedToken("[MASK]", lstrip=True, normalized=False),
        pad_token = "<pad>",
        unk_token = "<unk>",
        bos_token = "[CLS]",
        eos_token = "[SEP]",
        sep_token = "[SEP]",
        cls_token = "[CLS]",
        name_or_path="palibert",
        remove_space=True,
        keep_accents=True,
        do_lower_case=True,
        special_tokens_map_file=None
    )

    # Save
    if args.save_json:
        tokenizer.save(args.save_json)
    if args.save_model:
        os.makedirs(args.save_model, exist_ok=True)
        tokenizer.model.save(args.save_model)
    if args.save_as_pretrained:
        fast_tokenizer.save_pretrained(args.save_as_pretrained)


    # Test in stdout
    print("Here's the encoding of a couple sample catchphrases:")
    for catchphrase in PALI_CATCHPHRASES:
        print("\n*** " + catchphrase)
        print(tokenizer.encode(catchphrase).tokens)
        print(fast_tokenizer.encode(catchphrase))
        # print("> " + str(encoding.tokens))


if __name__ == "__main__":
    main()
