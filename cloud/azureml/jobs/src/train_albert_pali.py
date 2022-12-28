"""Train a BPE tokenizer on a Pali text from Sutta Central using HuggingFace."""
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
from transformers import (
    AutoTokenizer,
    AlbertConfig,
    AlbertForMaskedLM,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


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

LOGGER = logging.getLogger(__name__)


def train_tokenizer(args):
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

    LOGGER.info(f"tokenizer={tokenizer.to_str(pretty=True)}")

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
        name_or_path="palibert",
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

    LOGGER.info(f"fast_tokenizer={fast_tokenizer}")
    return fast_tokenizer


def get_model_config(args, tokenizer):
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
        max_position_embeddings=tokenizer.model_max_length,
        type_vocab_size=2,  # The vocabulary size of the token_type_ids passed when calling AlbertModel or TFAlbertModel ???
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        classifier_dropout_prob=0.1,
        position_embedding_type="absolute",  # relative_key, relative_key_query
        pad_token_id=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["pad_token"]),
        bos_token_id=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["bos_token"]),
        eos_token_id=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["eos_token"]),
    )

    LOGGER.info(f"base_config={base_config}")
    return base_config


def run(args):
    training_args = {}

    if os.path.isdir(args.output_dir):
        # try loading a checkpoint from the output (continued training if using lowpri)
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None:
            # there's no checkpoint in outputs
            LOGGER.info("No checkpoint found in --output_dir")
        else:
            # there's a checkpoint in outputs, using it instead
            LOGGER.info("Found a checkpoint in --output_dir, using it instead")
            args.checkpoint = args.output_dir
            training_args["resume_from_checkpoint"] = last_checkpoint
    else:
        # if dir does not exist, let's create it
        os.makedirs(args.output_dir)

    # TOKENIZER

    if args.checkpoint:
        # load existing tokenizer from checkpoint
        LOGGER.info("Loading tokenizer from --checkpoint")
        tokenizer_kwargs = {
            # "cache_dir": model_args.cache_dir,
            # "use_fast": model_args.use_fast_tokenizer,
            # "revision": model_args.model_revision,
            # "use_auth_token": True if model_args.use_auth_token else None,
        }
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, **tokenizer_kwargs)
        LOGGER.info(f"tokenizer={tokenizer}")
    else:
        # train tokenizer from scratch
        LOGGER.warning("Training tokenizer from scratch using --train_file")
        tokenizer = train_tokenizer(args)

    tokenizer.save_pretrained(args.output_dir)

    # MODEL

    if args.checkpoint:
        # load model from checkpoint
        LOGGER.info("Loading model from --checkpoint")
        # base_config = AlbertConfig.from_pretrained(args.checkpoint)
        # LOGGER.info(f"base_config={base_config}")
        model = AlbertForMaskedLM.from_pretrained(args.checkpoint)
        LOGGER.info(f"model_config={model.config}")
    else:
        LOGGER.warning("Training model from scratch")
        base_config = get_model_config(args, tokenizer)
        model = AlbertForMaskedLM(config=base_config)

    # TRAINING

    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=args.train_file,
        block_size=tokenizer.model_max_length,
    )
    # eval_dataset = LineByLineTextDataset(
    #     tokenizer=tokenizer,
    #     file_path="/home/ubuntu/data_local/wikitext-2-raw/wiki.test.raw",
    #     block_size=512,
    # )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=50,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        save_steps=2000,
        logging_steps=2000,
        learning_rate=1e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        # prediction_loss_only=True,
    )

    if args.do_train:
        trainer.train()

    trainer.save_model(args.output_dir)
    # eval_output = trainer.evaluate()
    # perplexity = math.exp(eval_output["eval_loss"])
    # print({"loss=eval_output["eval_loss"]})
    # result = {"perplexity=perplexity}
    # print(result)


def main():
    """Parse args and run script"""
    # create parser for cli args
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default=None,
        help="Path to pre-trained checkpoint(s), if None start from scratch using --train_file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        default=None,
        help="Path to save outputs of training",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to the file to train tokenizer and model.",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        required=False,
        default=None,
        help="Path to the file to evaluate model (default: None).",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=False,
        default=30000,
    )
    parser.add_argument(
        "--do_train",
        default=False,
        action="store_true",
        help="actually train the model",
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
    LOGGER.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    handler.setFormatter(log_format)
    LOGGER.addHandler(handler)

    # run function contains everything
    run(args)


if __name__ == "__main__":
    main()
