import os
import glob
import logging

from .types import SCDatasetPath, SCDataFilePath
from typing import List


def list_discoverable_sc_paths(sc_clone_root: str) -> List[SCDatasetPath]:
    all_paths = []

    all_paths.append(
        SCDatasetPath(
            directory=os.path.join(
                sc_clone_root,
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
        SCDatasetPath(
            directory=os.path.join(
                sc_clone_root,
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
        SCDatasetPath(
            directory=os.path.join(
                sc_clone_root,
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
        SCDatasetPath(
            directory=os.path.join(
                sc_clone_root,
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
        SCDatasetPath(
            directory=os.path.join(
                sc_clone_root,
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
        SCDatasetPath(
            directory=os.path.join(
                sc_clone_root,
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
        SCDatasetPath(
            directory=os.path.join(
                sc_clone_root,
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
        SCDatasetPath(
            directory=os.path.join(
                sc_clone_root,
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
        SCDatasetPath(
            directory=os.path.join(
                sc_clone_root,
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
        SCDatasetPath(
            directory=os.path.join(
                sc_clone_root,
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


def discover_all_sc_files(sc_clone_root: str) -> List[SCDataFilePath]:
    sc_file_entries = []
    for sc_dir in list_discoverable_sc_paths(sc_clone_root):
        files = glob.glob(
            os.path.join(sc_dir.directory, "**", "*.json"), recursive=True
        )

        for file in files:
            _entry = SCDataFilePath(
                file_path=file, lang=sc_dir.lang, corpus=sc_dir.corpus
            )

            sc_file_entries.append(_entry)

        logging.getLogger(__name__).info(
            f"discover_all_sc_files() found {len(files)} files in {sc_dir.directory} (lang={sc_dir.lang}, corpus={sc_dir.corpus})"
        )

    return sc_file_entries
