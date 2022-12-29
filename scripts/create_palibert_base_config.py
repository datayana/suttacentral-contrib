"""Creates the initial config for PaliBERT.

This script will:
- train an initial BPE tokenizer on a given line-by-line pali dataset
- create/adapt an initial Albert configuration
- export in a given directory
"""
import os
import sys
import argparse
import logging
import json

# imports for the tokenizer
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
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

# imports for the model
from transformers import AutoTokenizer, AlbertConfig, AlbertForMaskedLM


def run(args, logger):
    SPECIAL_TOKENS = dict(
        mask_token=AddedToken("[MASK]", lstrip=True, normalized=False),
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="[CLS]",
        eos_token="[SEP]",
        sep_token="[SEP]",
        cls_token="[CLS]",
    )
    SPECIAL_TOKENS_LIST = [
        "<pad>",
        "<unk>",
        "[CLS]",
        "[SEP]",
        AddedToken("[MASK]", lstrip=True, normalized=False),
    ]

    # Initialize
    tokenizer = Tokenizer(
        BPE(
            # vocab=None,
            # merges=None,
            # cache_capacity=None,
            # dropout=None,
            unk_token=SPECIAL_TOKENS["unk_token"],
            continuing_subword_prefix="#",
            # end_of_word_suffix=None,
            # fuse_unk=None,
        )
    )
    tokenizer.pre_tokenizer = Sequence(
        [
            Whitespace(),
            CharDelimiterSplit("\n"),
            Punctuation(),
        ]
    )
    tokenizer.normalizer = Lowercase()

    logger.info(f"tokenizer={tokenizer.to_str(pretty=True)}")

    # Create tokenizer trainer
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequence=2,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS_LIST,
    )

    # Train the tokenizer
    tokenizer.train([args.train_file], trainer)

    # add post processor
    tokenizer.post_processor = TemplateProcessing(
        single=f"{SPECIAL_TOKENS['bos_token']} $A {SPECIAL_TOKENS['eos_token']}",
        pair=f"{SPECIAL_TOKENS['bos_token']} $A {SPECIAL_TOKENS['eos_token']} $B:1 {SPECIAL_TOKENS['eos_token']}:1",
        special_tokens=[
            (
                SPECIAL_TOKENS["bos_token"],
                tokenizer.token_to_id(SPECIAL_TOKENS["bos_token"]),
            ),
            (
                SPECIAL_TOKENS["eos_token"],
                tokenizer.token_to_id(SPECIAL_TOKENS["eos_token"]),
            ),
        ],
    )

    # convert to transformers tokenizer
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=512,
        name_or_path="palibert-base",
        remove_space=True,
        keep_accents=True,
        do_lower_case=True,
        special_tokens_map_file=None,
        padding_side="right",
        truncation_side="right",
        rstrip=False,
        lstrip=True,
        single_word=False,
        normalized=False,
        **SPECIAL_TOKENS,
    )

    logger.info(f"fast_tokenizer={fast_tokenizer}")

    # save into the output dir
    fast_tokenizer.save_pretrained(args.output_dir)

    # create a model base config
    # see https://huggingface.co/docs/transformers/model_doc/albert#transformers.AlbertConfig
    # leaving no default values implicit
    base_config = AlbertConfig(
        vocab_size=args.vocab_size,
        embedding_size=128,
        hidden_size=768,  # 4096
        num_hidden_layers=12,
        num_hidden_groups=1,
        num_attention_heads=12,  # 64
        intermediate_size=3072,  # 16384
        inner_group_num=1,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=fast_tokenizer.model_max_length,
        type_vocab_size=1,  # The vocabulary size of the token_type_ids passed when calling AlbertModel or TFAlbertModel ???
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        classifier_dropout_prob=0.1,
        position_embedding_type="absolute",  # relative_key, relative_key_query
        pad_token_id=fast_tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["pad_token"]),
        bos_token_id=fast_tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["bos_token"]),
        eos_token_id=fast_tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["eos_token"]),
    )

    logger.info(f"base_config={base_config}")

    # save config
    base_config.to_json_file(os.path.join(args.output_dir, "config.json"))
    # model = AlbertForMaskedLM(config=base_config)
    # model.config.save_pretrained(args.output_dir)
    # model.config.to_json_file(os.path.join(args.output_dir, "config.json"))


def main():
    """Parse args and run script"""
    # create parser for cli args
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        default=None,
        help="Path to save tokenizer and model configs",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to the file to train tokenizer and model.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=False,
        default=30000,
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
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    handler.setFormatter(log_format)
    logger.addHandler(handler)

    # create output dir if necessary
    os.makedirs(args.output_dir, exist_ok=True)

    # run does everything
    run(args, logger)


if __name__ == "__main__":
    main()
