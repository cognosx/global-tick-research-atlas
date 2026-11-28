# Auto-generated mirror of notebooks/01_corpus/05_validation_screening.ipynb

# # 05 · Validation of the screening: adjudication + corpus-weighted metrics
# 
# The exact steps used on 2026-11-28 (code: `src/gtra/validation.py`):
# 1. load your blind codes (`gtra_validation_coding.xlsx`) and the model decisions
# 2. quick unweighted comparison
# 3. write the adjudication sheet (include/exclude disagreements only)
# 4. read your `final_class` decisions back
# 5. metrics weighted to the corpus with the sampling weights + stratified bootstrap 95% intervals
# 6. save labels (no abstracts) and the metrics table

from google.colab import drive; drive.mount('/content/drive')
pass  # shell: pip -q install pandas pyarrow openpyxl
import os, sys, glob
if not os.path.exists('/content/gtra'):
    pass  # shell: git clone -q https://github.com/cognosx/global-tick-research-atlas /content/gtra
else:
    pass  # shell: git -C /content/gtra pull -q
sys.path.insert(0, '/content/gtra/src')
import importlib, pandas as pd
from pathlib import Path
import gtra.validation as V; importlib.reload(V)
pd.set_option('display.max_columns', None)

ROOT   = "/content/drive/Othercomputers/My laptop/D:/COGNOS X bon/TICKS/ticks_research"   # <-- yours
RUN    = "openalex_2026-11-26"
SCREEN = Path(f"{ROOT}/preprocessing_data/{RUN}/screening")
SHEET  = SCREEN / "gtra_validation_coding.xlsx"
ADJ    = SCREEN / "gtra_validation_adjudication.xlsx"     # or a .csv exported from Google Sheets

parts = sorted(glob.glob(str(SCREEN / "full" / "*" / "parts" / "part_*.parquet")))
codes = pd.concat([pd.read_parquet(p) for p in parts]).drop_duplicates("oa_id", keep="last")
m = V.load_validation(SHEET, codes)
print(len(m), "coded records")
V.score_raw(m, ref="h0")

# Adjudication sheet (written once; fill final_class, then save it as ADJ)
if not ADJ.exists():
    V.write_adjudication(m, ADJ)

# After you filled final_class
m = V.apply_adjudication(m, ADJ)
rep = V.report(m, B=1000)
for k, v in rep.items():
    print("\n##", k); print(v.to_string(index=False))

# Save: labels for the repo (no abstracts) + metrics table for the methods
V.save_labels(m, SCREEN / "gtra_v1_validation_labels.parquet")
pd.concat([v.assign(table=k) for k, v in rep.items() if k != "unweighted"]).to_csv(SCREEN / "gtra_v1_validation_metrics.csv", index=False)
print("saved to", SCREEN)
