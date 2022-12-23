import os
import sys
import logging
import suttacentral.contrib as sc
from suttacentral.contrib import discover

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(log_format)
logger.addHandler(handler)

sc_root = os.path.join(os.path.dirname(__file__), "..", "sc-data")
sc.configure(SC_DATA_ROOT=sc_root)

sc_flat_entries = discover.process_all_sc_files()

with open("data/pali_for_bpe.txt", "w", encoding="utf-8") as out_file:
    for entry in sc_flat_entries:
        if entry.pali and not entry.paragraph_id.startswith("0"):
            try:
                int(entry.pali)
                continue
            except:
                pass
            out_file.write(entry.pali)
            out_file.write("\n")
