import os
import logging
from dataclasses import dataclass

# location to the sc-data clone
@dataclass
class SuttaCentralContribConfig:
    SC_DATA_ROOT: str = None
    SC_DATA_PLI_MS_SUTTA: str = None
    SC_DATA_EN_SUTTA_SUJATO: str = None


_DEFAULTS = SuttaCentralContribConfig()


def configure(
    SC_DATA_ROOT=None, SC_DATA_PLI_MS_SUTTA=None, SC_DATA_EN_SUTTA_SUJATO=None
):
    global _DEFAULTS

    if SC_DATA_ROOT is not None:
        if not os.path.exists(SC_DATA_ROOT):
            raise FileNotFoundException(
                f"SC_DATA_ROOT directory {SC_DATA_ROOT} doesn't exist"
            )
        if not os.path.isdir(SC_DATA_ROOT):
            raise ValueError(f"SC_DATA_ROOT={SC_DATA_ROOT} is not a directory")

        _DEFAULTS.SC_DATA_ROOT = SC_DATA_ROOT

        if SC_DATA_PLI_MS_SUTTA is None:
            SC_DATA_PLI_MS_SUTTA = os.path.join(
                SC_DATA_ROOT, "sc_bilara_data", "root", "pli", "ms", "sutta"
            )

        if SC_DATA_EN_SUTTA_SUJATO is None:
            SC_DATA_EN_SUTTA_SUJATO = os.path.join(
                SC_DATA_ROOT, "sc_bilara_data", "translation", "en", "sujato", "sutta"
            )

    if SC_DATA_PLI_MS_SUTTA is not None:
        if not os.path.exists(SC_DATA_PLI_MS_SUTTA):
            raise FileNotFoundException(
                f"SC_DATA_PLI_MS_SUTTA directory {SC_DATA_PLI_MS_SUTTA} doesn't exist"
            )
        if not os.path.isdir(SC_DATA_PLI_MS_SUTTA):
            raise ValueError(
                f"SC_DATA_PLI_MS_SUTTA={SC_DATA_PLI_MS_SUTTA} is not a directory"
            )

        _DEFAULTS.SC_DATA_PLI_MS_SUTTA = SC_DATA_PLI_MS_SUTTA

    if SC_DATA_EN_SUTTA_SUJATO is not None:
        if not os.path.exists(SC_DATA_EN_SUTTA_SUJATO):
            raise FileNotFoundException(
                f"SC_DATA_EN_SUTTA_SUJATO directory {SC_DATA_EN_SUTTA_SUJATO} doesn't exist"
            )
        if not os.path.isdir(SC_DATA_EN_SUTTA_SUJATO):
            raise ValueError(
                f"SC_DATA_EN_SUTTA_SUJATO={SC_DATA_EN_SUTTA_SUJATO} is not a directory"
            )

        _DEFAULTS.SC_DATA_EN_SUTTA_SUJATO = SC_DATA_EN_SUTTA_SUJATO

    logging.getLogger(__name__).info(f"Configured {__name__} with {_DEFAULTS}")
