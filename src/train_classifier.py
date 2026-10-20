"""Melatih model klasifikasi penyebab deforestasi dan mengekspor hasil untuk dashboard."""
import json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict, train_test_split

ROOT = Path(__file__).parent.parent
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "gangguan_hutan.csv"
FITUR = ["luas_ha", "kekompakan_bentuk", "indikasi_kebakaran", "indikasi_tambang",
         "jarak_jalan_km", "jarak_lahan_tani_km", "jarak_tambang_km", "kemiringan_derajat"]
LABEL = "penyebab_sebenarnya"

df = pd.read_csv(CSV)
X, y = df[FITUR], df[LABEL]
model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")

# evaluasi: hold-out 25%
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=42)
model.fit(Xtr, ytr)
pred_te = model.predict(Xte)
acc, f1 = accuracy_score(yte, pred_te), f1_score(yte, pred_te, average="macro")
print(classification_report(yte, pred_te, zero_division=0))
kelas = sorted(y.unique())
print("Confusion matrix (baris=aktual, kolom=prediksi):", kelas)
print(confusion_matrix(yte, pred_te, labels=kelas))

# prediksi seluruh data lewat cross-validation agar tabel dashboard tidak bias
proba = cross_val_predict(model, X, y, cv=5, method="predict_proba")
df["prediksi"] = np.array(kelas)[proba.argmax(axis=1)]
df["keyakinan"] = np.round(proba.max(axis=1) * 100).astype(int)

model.fit(X, y)
joblib.dump(model, ROOT / "models" / "forestwatch_rf.joblib")

hasil = {
  "meta": {"akurasi": round(acc * 100, 1), "f1_macro": round(f1 * 100, 1), "jumlah_data": len(df)},
  "fitur_penting": sorted(
      [{"nama": f, "nilai": round(float(v), 4)} for f, v in zip(FITUR, model.feature_importances_)],
      key=lambda d: -d["nilai"]),
  "baris": df[["id_area", "provinsi", "kabupaten", "luas_ha", "kekompakan_bentuk", "indikasi_kebakaran",
               "indikasi_tambang", "prediksi", "keyakinan"]].to_dict("records"),
}
out = ROOT / "docs" / "data" / "hasil.js"
out.write_text("window.FORESTWATCH = " + json.dumps(hasil, ensure_ascii=False) + ";\n", encoding="utf-8")
df.to_csv(ROOT / "data" / "hasil_klasifikasi.csv", index=False)
print(f"\nAkurasi hold-out {acc:.1%} | F1 macro {f1:.1%} | dashboard diperbarui: {out}")
