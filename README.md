# ForestWatch AI

Sistem klasifikasi penyebab deforestasi. Dari data gangguan hutan (luas, bentuk area, indikator api dan tambang, jarak ke jalan/lahan tani/tambang, kemiringan), model machine learning menebak penyebabnya: **pertanian, pertambangan, logging, kebakaran, atau gangguan alami**. Hasilnya tampil di dashboard web.

> Dataset di repo ini **sintetis** untuk keperluan belajar dan demo, bukan data pemantauan nyata.

![Dashboard ForestWatch AI](docs/screenshots/dashboard.png)

## Fitur
1. **Input data gangguan hutan**: file CSV di `data/` (ID area, lokasi, luas, bentuk, indikator kebakaran/tambang, dan fitur lingkungan).
2. **Klasifikasi dengan Python**: Random Forest (scikit-learn), dievaluasi dengan hold-out 25% dan 5-fold cross-validation.
3. **Dashboard analisis**: jumlah kasus, total luas, proporsi luas per penyebab, fitur paling berpengaruh, serta tabel hasil yang bisa dicari, difilter, dan diurutkan.
4. **Dokumentasi**: kode, dataset contoh, panduan instalasi, dan screenshot.

## Struktur repository
```
forestwatch-ai/
├── data/
│   ├── gangguan_hutan.csv        dataset contoh (400 area)
│   └── hasil_klasifikasi.csv     keluaran model
├── src/
│   ├── generate_data.py          membuat dataset sintetis
│   ├── train_classifier.py       latih model, evaluasi, ekspor ke dashboard
│   └── predict.py                prediksi untuk CSV baru
├── models/forestwatch_rf.joblib  model terlatih
├── docs/                         dashboard (HTML, CSS, JS) untuk GitHub Pages
│   ├── index.html, style.css, app.js
│   ├── data/hasil.js             data hasil klasifikasi untuk dashboard
│   └── screenshots/
├── requirements.txt
└── README.md
```

## Instalasi
```bash
git clone https://github.com/<username>/forestwatch-ai.git
cd forestwatch-ai
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Cara pakai
```bash
python src/generate_data.py      # opsional: buat ulang dataset contoh
python src/train_classifier.py   # latih model dan perbarui dashboard
python src/predict.py data/area_baru.csv   # prediksi data baru (tanpa kolom penyebab_sebenarnya)
```
Untuk memakai data sendiri, samakan nama kolom dengan `data/gangguan_hutan.csv`, lalu jalankan `python src/train_classifier.py path/ke/datamu.csv`.

Dashboard dibuka dengan membuka `docs/index.html` di browser, atau lewat server lokal: `python -m http.server -d docs 8000`.

## Hosting di GitHub Pages
1. Push repo ke GitHub.
2. Buka **Settings → Pages**.
3. Pada *Build and deployment*, pilih **Deploy from a branch**, branch `main`, folder **/docs**, lalu Save.
4. Dashboard aktif di `https://<username>.github.io/forestwatch-ai/`.

## Hasil model (data contoh)
Akurasi hold-out 98%, F1 macro 98%. Angka setinggi ini wajar karena data sintetis dibuat dengan pola yang jelas. Pada data nyata, hasilnya akan lebih rendah dan perlu validasi lapangan.

## Pengembangan lanjutan
- Mengganti data sintetis dengan data deteksi gangguan nyata (misalnya citra satelit dan titik panas).
- Menambah fitur spasial dan temporal, lalu membandingkan model lain (Gradient Boosting, dsb.).
- Menambah peta sebaran area.

## Lisensi
MIT (silakan sesuaikan).
