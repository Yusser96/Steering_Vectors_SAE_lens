import os
import pandas as pd
from collections import defaultdict
import json

base_dir = "az_cy_de_en_he_id_backtranslated_and_synthetic_data"
new_data = defaultdict(list)
for file in os.listdir(base_dir):
    if file.endswith("_human_back.csv"):
        df = pd.read_csv(os.path.join(base_dir,file))
        new_data["good"].extend(df["original_text"])
        new_data["bad"].extend(df["backtranslated_text"])

import json 
with open("data/good_bad_data.json", "w" , encoding="utf8") as f:
    json.dump(new_data,f, indent=4, ensure_ascii=False)