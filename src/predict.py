"""Prediksi penyebab untuk CSV baru (tanpa kolom label).
Pemakaian: python src/predict.py data/area_baru.csv"""
import sys
from pathlib import Path
import joblib, pandas as pd

ROOT = Path(__file__).parent.parent
model = joblib.load(ROOT / "models" / "forestwatch_rf.joblib")
df = pd.read_csv(sys.argv[1])
df["prediksi"] = model.predict(df[list(model.feature_names_in_)])
out = Path(sys.argv[1]).with_name("prediksi_" + Path(sys.argv[1]).name)
df.to_csv(out, index=False)
print(df[["id_area", "prediksi"]].to_string(index=False)); print("Disimpan:", out)
