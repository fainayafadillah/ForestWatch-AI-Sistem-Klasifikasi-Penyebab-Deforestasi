"""Membuat dataset contoh gangguan hutan (data sintetis untuk demo)."""
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N = 400

# profil tiap penyebab: luas (median ha), kekompakan bentuk (alpha,beta), peluang api & tambang,
# jarak ke jalan, ke lahan tani, ke tambang (km), kemiringan (derajat)
P = {
 "Pertanian":    dict(luas=14,  shp=(6,3), api=.12, tbg=.02, jln=3,  tni=1.0, tam=40, lrg=6),
 "Pertambangan": dict(luas=45,  shp=(3,3), api=.03, tbg=.92, jln=6,  tni=15,  tam=1.0, lrg=12),
 "Logging":      dict(luas=22,  shp=(2,5), api=.05, tbg=.03, jln=1.5,tni=10,  tam=35, lrg=18),
 "Kebakaran":    dict(luas=120, shp=(2,2), api=.95, tbg=.04, jln=7,  tni=4,   tam=30, lrg=5),
 "Alami":        dict(luas=5,   shp=(2,2), api=.04, tbg=.01, jln=15, tni=22,  tam=60, lrg=28),
}
W = {"Pertanian":.30,"Pertambangan":.15,"Logging":.22,"Kebakaran":.20,"Alami":.13}
LOK = [("Bengkulu","Bengkulu Utara"),("Bengkulu","Kaur"),("Jambi","Merangin"),("Riau","Pelalawan"),
       ("Sumatera Selatan","Musi Banyuasin"),("Kalimantan Barat","Ketapang"),("Kalimantan Tengah","Kotawaringin Timur"),
       ("Kalimantan Timur","Kutai Kartanegara"),("Sulawesi Tengah","Morowali"),("Papua Barat","Teluk Bintuni")]

rows = []
for i in range(N):
    k = rng.choice(list(P), p=list(W.values())); p = P[k]
    prov, kab = LOK[rng.integers(len(LOK))]
    rows.append(dict(
        id_area=f"FW-{i+1:04d}", provinsi=prov, kabupaten=kab,
        luas_ha=round(float(rng.lognormal(np.log(p["luas"]), .55)), 1),
        kekompakan_bentuk=round(float(np.clip(rng.beta(*p["shp"]), .05, .99)), 2),
        indikasi_kebakaran=int(rng.random() < p["api"]),
        indikasi_tambang=int(rng.random() < p["tbg"]),
        jarak_jalan_km=round(float(rng.gamma(2, p["jln"]/2)), 1),
        jarak_lahan_tani_km=round(float(rng.gamma(2, p["tni"]/2)), 1),
        jarak_tambang_km=round(float(rng.gamma(2, p["tam"]/2)), 1),
        kemiringan_derajat=round(float(np.clip(rng.normal(p["lrg"], 5), 0, 45)), 1),
        penyebab_sebenarnya=k))
out = Path(__file__).parent.parent / "data" / "gangguan_hutan.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("Dataset tersimpan:", out, f"({N} baris)")
