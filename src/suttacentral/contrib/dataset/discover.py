import os
import logging
import traceback
from tqdm import tqdm

from .paths import discover_all_sc_files
from .types import SCTextData
from typing import List
from .parsing import parse_sc_file


def sc_get_flat_entries(sc_clone_root: str) -> List[SCTextData]:
    sc_text_entries = []

    all_files = discover_all_sc_files(sc_clone_root)

    logging.getLogger(__name__).info(f"Discovered {len(all_files)} files total.")

    for sc_file in tqdm(all_files):
        sc_text_entries.extend(parse_sc_file(sc_file))

    logging.getLogger(__name__).info(f"Parsed {len(sc_text_entries)} entries total.")

    return sc_text_entries


def group_entries_per_id(sc_entries: List[SCTextData]):
    keys_in_join = {}
    keys_remaining_pli = {}
    keys_remaining_en = {}

    for entry in sc_entries:
        if entry.pali is not None:
            if entry.id in keys_remaining_en:
                # join and remove from en
                entry.english = keys_remaining_en[entry.id].english
                keys_in_join[entry.id] = entry
                del keys_remaining_en[entry.id]
            else:
                keys_remaining_pli[entry.id] = entry
        elif entry.english is not None:
            if entry.id in keys_remaining_pli:
                # join and remove from pli
                entry.pali = keys_remaining_pli[entry.id].pali
                keys_in_join[entry.id] = entry
                del keys_remaining_pli[entry.id]
            else:
                keys_remaining_en[entry.id] = entry
        else:
            raise ValueError(
                f"Entry has neither english or pali content entry={entry}."
            )

    logging.getLogger(__name__).info(
        f"After join, we have {len(keys_in_join)} pali-english pairs."
    )
    if keys_remaining_pli:
        logging.getLogger(__name__).warning(
            f"After join, {len(keys_remaining_pli)} pali entries have no english correspondance."
        )
    if keys_remaining_en:
        logging.getLogger(__name__).warning(
            f"After join, {len(keys_remaining_en)} english entries have no pali correspondance."
        )

    return keys_in_join, keys_remaining_pli, keys_remaining_en
