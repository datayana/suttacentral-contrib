import os
import sys
import argparse
import logging
import re

from suttacentral.contrib.dataset.discover import sc_get_flat_entries


def main():
    """Parse args and run script"""
    # create parser for cli args
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--sc_root_clone",
        type=str,
        required=True,
        help="Path to the suttacentral/sc-data clone root.",
    )
    parser.add_argument(
        "--export_train_file",
        type=str,
        required=True,
        help="Path to store the file to train BPE.",
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

    sc_flat_entries = sc_get_flat_entries(args.sc_root_clone)

    logger.info(f"Saving data to {args.export_train_file}")
    with open(args.export_train_file, "w", encoding="utf-8") as out_file:
        for entry in sc_flat_entries:
            # pass titles or subtitles
            if entry.paragraph_id.startswith("0"):
                continue

            # pass if there's no pali
            if entry.pali is None:
                continue

            pali_text = entry.pali.strip()

            # if just number or combo (ex: 123-125)
            if re.match("[0-9\-]+", pali_text):
                continue

            # remove "[N]"
            pali_text = re.sub(r"\[[0-9]+\]", "", pali_text)

            # remove training ".N"
            pali_text = re.sub(r" \.[0-9]+$", "", pali_text)

            # if after that, line is empty, pass
            if not pali_text:
                continue

            out_file.write(pali_text)
            out_file.write("\n")


if __name__ == "__main__":
    main()
