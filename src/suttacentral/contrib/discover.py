import os
import sys
import glob
import logging
import traceback
import json
from dataclasses import dataclass
from tqdm import tqdm

from suttacentral.contrib.config import _DEFAULTS


def hello_world():
    print("hello from the other side")
    print(_DEFAULTS.SC_DATA_ROOT)
    print(_DEFAULTS.SC_DATA_PLI_MS_SUTTA)


@dataclass
class SCPathEntry:
    directory: str
    lang: str
    corpus: str


def list_discoverable_sc_paths():
    all_paths = []

    all_paths.append(
        SCPathEntry(
            directory=os.path.join(
                _DEFAULTS.SC_DATA_ROOT,
                "sc_bilara_data",
                "root",
                "pli",
                "ms",
                "sutta",
                "an",
            ),
            lang="pli",
            corpus="sutta/an",
        )
    )
    all_paths.append(
        SCPathEntry(
            directory=os.path.join(
                _DEFAULTS.SC_DATA_ROOT,
                "sc_bilara_data",
                "root",
                "pli",
                "ms",
                "sutta",
                "dn",
            ),
            lang="pli",
            corpus="sutta/dn",
        )
    )
    all_paths.append(
        SCPathEntry(
            directory=os.path.join(
                _DEFAULTS.SC_DATA_ROOT,
                "sc_bilara_data",
                "root",
                "pli",
                "ms",
                "sutta",
                "kn",
            ),
            lang="pli",
            corpus="sutta/kn",
        )
    )
    all_paths.append(
        SCPathEntry(
            directory=os.path.join(
                _DEFAULTS.SC_DATA_ROOT,
                "sc_bilara_data",
                "root",
                "pli",
                "ms",
                "sutta",
                "mn",
            ),
            lang="pli",
            corpus="sutta/mn",
        )
    )
    all_paths.append(
        SCPathEntry(
            directory=os.path.join(
                _DEFAULTS.SC_DATA_ROOT,
                "sc_bilara_data",
                "root",
                "pli",
                "ms",
                "sutta",
                "sn",
            ),
            lang="pli",
            corpus="sutta/sn",
        )
    )

    all_paths.append(
        SCPathEntry(
            directory=os.path.join(
                _DEFAULTS.SC_DATA_ROOT,
                "sc_bilara_data",
                "translation",
                "en",
                "sujato",
                "sutta",
                "an",
            ),
            lang="en",
            corpus="sutta/an",
        )
    )
    all_paths.append(
        SCPathEntry(
            directory=os.path.join(
                _DEFAULTS.SC_DATA_ROOT,
                "sc_bilara_data",
                "translation",
                "en",
                "sujato",
                "sutta",
                "dn",
            ),
            lang="en",
            corpus="sutta/dn",
        )
    )
    all_paths.append(
        SCPathEntry(
            directory=os.path.join(
                _DEFAULTS.SC_DATA_ROOT,
                "sc_bilara_data",
                "translation",
                "en",
                "sujato",
                "sutta",
                "kn",
            ),
            lang="en",
            corpus="sutta/kn",
        )
    )
    all_paths.append(
        SCPathEntry(
            directory=os.path.join(
                _DEFAULTS.SC_DATA_ROOT,
                "sc_bilara_data",
                "translation",
                "en",
                "sujato",
                "sutta",
                "mn",
            ),
            lang="en",
            corpus="sutta/mn",
        )
    )
    all_paths.append(
        SCPathEntry(
            directory=os.path.join(
                _DEFAULTS.SC_DATA_ROOT,
                "sc_bilara_data",
                "translation",
                "en",
                "sujato",
                "sutta",
                "sn",
            ),
            lang="en",
            corpus="sutta/sn",
        )
    )

    logging.getLogger(__name__).info(f"SC has {len(all_paths)} discoverable paths")

    return all_paths


@dataclass
class SCFileEntry:
    file_path: str
    lang: str
    corpus: str


def discover_all_sc_files():
    sc_file_entries = []
    for sc_dir in list_discoverable_sc_paths():
        files = glob.glob(
            os.path.join(sc_dir.directory, "**", "*.json"), recursive=True
        )

        for file in files:
            _entry = SCFileEntry(file_path=file, lang=sc_dir.lang, corpus=sc_dir.corpus)

            sc_file_entries.append(_entry)

        logging.getLogger(__name__).info(
            f"discover_all_sc_files() found {len(files)} files in {sc_dir.directory} (lang={sc_dir.lang}, corpus={sc_dir.corpus})"
        )

    return sc_file_entries


@dataclass
class SCPaliEnglishEntry:
    id: str
    corpus_id: str
    text_id: str
    paragraph_id: str
    pali: str = None
    english: str = None


def parse_sc_file_entries(file_path: str, lang: str, corpus: str) -> list:
    if lang not in ["en", "pli"]:
        raise ValueError("lang can be either en or pli... for now.")

    if not os.path.exists(file_path):
        raise FileNotFoundException(f"file_path={file_path} cannot be found")
    if not os.path.isfile(file_path):
        raise ValueError("file_path={file_path} has to be the path to a file")

    entries = []

    with open(file_path, "r", encoding="utf-8") as in_file:
        file_content = in_file.read()
        if len(file_content) == 0:
            raise Exception(f"file at path {file_path} is empty")

        try:
            json_content = json.loads(file_content)
        except Exception as e:
            raise Exception(
                f"Could not parse json content from file {file_path}: {traceback.format_exc()}"
            )

        for k in json_content:
            if not isinstance(k, str):
                raise Exception(f"In file {file_path}, key {k} is not a string")
            if not isinstance(json_content[k], str):
                raise Exception(
                    f"In file {file_path}, key={k} should contain a string, instead contains {json_content[k]}"
                )

            _entry_id = k
            _text_id, _paragraph_id = tuple(k.split(":"))

            _entry = SCPaliEnglishEntry(
                id=_entry_id,
                corpus_id=corpus,
                text_id=_text_id,
                paragraph_id=_paragraph_id,
                pali=(json_content[k] if lang == "pli" else None),
                english=(json_content[k] if lang == "en" else None),
            )

            entries.append(_entry)

        logging.getLogger(__name__).debug(
            f"Parsed {len(entries)} entries from file {file_path}."
        )

    return entries


def process_all_sc_files():
    sc_text_entries = []

    all_files = discover_all_sc_files()

    logging.getLogger(__name__).info(f"Discovered {len(all_files)} files total.")

    for sc_file in tqdm(all_files):
        sc_text_entries.extend(
            parse_sc_file_entries(
                sc_file.file_path, lang=sc_file.lang, corpus=sc_file.corpus
            )
        )

    logging.getLogger(__name__).info(f"Parsed {len(sc_text_entries)} entries total.")

    return sc_text_entries


def join_entries_per_lang(sc_entries):
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
