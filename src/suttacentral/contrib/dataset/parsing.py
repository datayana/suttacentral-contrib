import os
import logging
import traceback
import json
from .types import SCDataFilePath
from .types import SCTextData
from typing import List
import re


def parse_sc_file(sc_file: SCDataFilePath) -> List[SCTextData]:
    if sc_file.lang not in ["en", "pli"]:
        raise ValueError("lang can be either en or pli... for now.")

    if not os.path.exists(sc_file.file_path):
        raise FileNotFoundException(f"file_path={sc_file.file_path} cannot be found")
    if not os.path.isfile(sc_file.file_path):
        raise ValueError(f"file_path={sc_file.file_path} has to be the path to a file")

    entries = []

    with open(sc_file.file_path, "r", encoding=sc_file.encoding) as in_file:
        file_content = in_file.read()
        if len(file_content) == 0:
            raise Exception(f"file at path {sc_file.file_path} is empty")

        try:
            json_content = json.loads(file_content)
        except Exception as e:
            raise Exception(
                f"Could not parse json content from file {sc_file.file_path}: {traceback.format_exc()}"
            )

        for k in json_content:
            if not isinstance(k, str):
                raise Exception(f"In file {sc_file.file_path}, key {k} is not a string")
            if not isinstance(json_content[k], str):
                raise Exception(
                    f"In file {sc_file.file_path}, key={k} should contain a string, instead contains {json_content[k]}"
                )

            _entry_id = k
            _text_id, _paragraph_id = tuple(k.split(":"))

            _entry = SCTextData(
                id=_entry_id,
                corpus_id=sc_file.corpus,
                text_id=_text_id,
                paragraph_id=_paragraph_id,
                pali=(json_content[k] if sc_file.lang == "pli" else None),
                english=(json_content[k] if sc_file.lang == "en" else None),
            )

            entries.append(_entry)

        logging.getLogger(__name__).debug(
            f"Parsed {len(entries)} entries from file {sc_file.file_path}."
        )

    return entries
