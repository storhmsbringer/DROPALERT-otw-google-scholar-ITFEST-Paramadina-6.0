"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DROPALERT — Deteksi Risiko Putus Sekolah & Pemetaan Klaster Provinsi        ║
║  Dashboard Interaktif — direvisi agar konsisten dengan                        ║
║  sahda_DropAlert_Final.ipynb & dropalert_cluster_provinsi.csv                 ║
║  Sumber Data: BPS 2021–2025                                                   ║
║  Jalankan: streamlit run app.py                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

CATATAN REVISI PENTING (baca sebelum mengedit lebih lanjut)
-------------------------------------------------------------------------------
File ini ditulis ulang dari app.py versi lama agar seluruh definisi fitur,
model, metrik, dan hasil klastering PERSIS mengikuti notebook final
(`sahda_DropAlert_Final.ipynb`) dan CSV hasil klastering terbaru
(`dropalert_cluster_provinsi.csv`). Perubahan utama dibanding versi lama:

1. Rasio murid-guru & murid-sekolah dibalik ke konvensi notebook baru
   (murid per guru / murid per sekolah — BUKAN guru/murid seperti versi lama).
2. Klastering provinsi TIDAK dihitung ulang secara live dengan subset fitur
   yang berbeda dari notebook. Dashboard ini memuat langsung
   `dropalert_cluster_provinsi.csv` sebagai single source of truth, memakai
   ke-14 fitur & label cluster yang sama persis dengan notebook.
3. Tabel evaluasi model (regresi & klasifikasi) memakai angka asli hasil
   `regresi_results` / `klasifikasi_results` dari notebook — bukan angka
   simulasi/versi lama.
4. "Model terbaik per target" ditentukan memakai aturan eksplisit notebook
   sendiri (R2 test > 0.5, lalu F1 tertinggi), bukan tabel lama yang tidak
   konsisten dengan notebook manapun.
5. Fitur "Prediksi Risiko" interaktif kini melatih ulang model PERSIS sesuai
   notebook (hiperparameter, split waktu 2021–2024 vs 2025, thresholding
   median dari data train) — dan HANYA aktif jika dataset mentah
   `Dataset_Gabungan_Fix.xlsx` tersedia/diunggah, supaya dashboard tidak
   pernah menampilkan prediksi dari model "asal-asalan" yang tidak bisa
   dipertanggungjawabkan ke notebook.
6. KPI, feature-importance, dan kategori risiko yang dulunya angka fiktif
   (mis. "Akurasi Model RF: 93.44%") sudah dihapus/diganti angka nyata.

Lihat expander "Catatan Metodologis" di halaman Evaluasi Model untuk
keterbatasan yang **belum** diperbaiki di notebook (ukuran test set kecil,
IQR-capping dihitung dari seluruh data, dsb.) — dashboard ini melaporkan
apa adanya, bukan menyembunyikannya.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════
# 1. KONFIGURASI HALAMAN
# ════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DropAlert | Deteksi Risiko Putus Sekolah",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════
# 2. CUSTOM CSS
# ════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp, p, h1, h2, h3, h4, h5, h6, label, li {
        font-family: 'Inter', sans-serif;
    }

    .stApp { background-color: #0E1117; color: #FFFFFF; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #121212 0%, #1E1E1E 80%, #4B0000 100%);
        border-right: 1px solid #2D2D2D;
    }

    .hero-container {
        background: linear-gradient(135deg, #800000 0%, #C0392B 45%, #E67E22 100%);
        padding: 40px 50px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.55);
        margin-bottom: 28px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .hero-container h1 { color: #fff; font-size: 1.7rem; line-height: 1.4; margin: 0 auto; font-weight: 700; text-align: center; }
    .hero-container p  { color: #FFE0CC; font-size: 0.92rem; margin: 10px auto 0; font-weight: 400; text-align: center; max-width: 780px; }

    .kpi-card {
        background: #1A1C23;
        border: 1px solid #2D2D2D;
        border-radius: 8px;
        padding: 16px 18px;
        text-align: center;
        transition: border-color .25s, transform .25s;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
    }
    .kpi-card:hover { border-color: #E74C3C; transform: translateY(-4px); }
    .kpi-card .label { color: #AAA; font-size: 0.72rem; letter-spacing: 0.5px; text-transform: uppercase; font-weight: 600; text-align: center; width: 100%;}
    .kpi-card .value { color: #FFF; font-size: 1.6rem; font-weight: 700; margin: 6px 0 0; text-align: center; width: 100%;}
    .kpi-card .sub { color: #888; font-size: 0.68rem; margin-top: 4px; }

    .section-title {
        color: #E74C3C;
        border-left: 4px solid #F39C12;
        padding-left: 14px;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 28px 0 14px;
    }

    div[data-testid="stButton"] > button {
        background: linear-gradient(90deg, #C0392B, #E67E22);
        color: #fff;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        width: 100%;
        padding: 12px;
        font-size: 1rem;
        transition: opacity .2s;
    }
    div[data-testid="stButton"] > button:hover { opacity: .88; color: #fff; }

    .badge-rendah  { background: #1E8449; color: #fff; padding: 6px 18px; border-radius: 4px; font-weight: 600; display: inline-block; }
    .badge-sedang  { background: #B7770D; color: #fff; padding: 6px 18px; border-radius: 4px; font-weight: 600; display: inline-block; }
    .badge-tinggi  { background: #C0392B; color: #fff; padding: 6px 18px; border-radius: 4px; font-weight: 600; display: inline-block; }

    .result-box {
        background: #1A1C23;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 25px;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700; font-size: 2.0rem !important; }
    div[data-testid="stMetricLabel"] { color: #AAAAAA !important; font-size: 0.9rem !important; margin-bottom: 5px; }
    div[data-testid="stMetricDelta"] { font-size: 1rem !important; }

    .insight-box {
        background: #14171E;
        border-left: 4px solid #E67E22;
        border-radius: 4px;
        padding: 18px 22px;
        margin-top: 16px;
        color: #E0E0E0;
        font-size: 0.93rem;
        line-height: 1.65;
    }
    .caveat-box {
        background: #14171E;
        border-left: 4px solid #566573;
        border-radius: 4px;
        padding: 14px 18px;
        margin-top: 12px;
        color: #BFC9CA;
        font-size: 0.85rem;
        line-height: 1.55;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# 3. KONSTANTA — PERSIS MENGIKUTI NOTEBOOK & CSV
# ════════════════════════════════════════════════════════

# 14 fitur X — nama & arah rasio PERSIS sama dengan notebook (murid per guru,
# murid per sekolah — bukan kebalikannya).
FITUR = [
    'gabungan_pendudukmiskin',
    'TPT',
    'NEET_usiamuda',
    'tenagakerjaformal',
    'gabungan_HLS',
    'gabungan_RLS',
    'rasio_murid_guru_SMA',
    'rasio_murid_guru_SMK',
    'rasio_murid_guru_SD',
    'rasio_murid_guru_SMP',
    'rasio_murid_sekolah_SMA',
    'rasio_murid_sekolah_SMK',
    'rasio_murid_sekolah_SD',
    'rasio_murid_sekolah_SMP',
]

# Fitur yang dipakai untuk K-Means (identik dengan FITUR di notebook cell 25)
CLUSTER_FEATURES = list(FITUR)

TARGET_LIST = ['ARPS_07to12', 'ARPS_13to15', 'ARPS_16to18', 'ARPS_19to23']

TARGET_LABELS = {
    'ARPS_07to12': '7–12 Tahun (SD)',
    'ARPS_13to15': '13–15 Tahun (SMP)',
    'ARPS_16to18': '16–18 Tahun (SMA)',
    'ARPS_19to23': '19–23 Tahun (Perguruan Tinggi)',
}

APS_COL = {
    'ARPS_07to12': 'APS_07to12',
    'ARPS_13to15': 'APS_13to15',
    'ARPS_16to18': 'APS_16to18',
    'ARPS_19to23': 'APS_19to23',
}

# Model & hiperparameter terbaik per target — dipilih memakai ATURAN EKSPLISIT
# notebook sendiri (markdown cell "Pemilihan Model Terbaik per Target"):
#   1) buang model dengan r2_test <= 0.5 (gagal generalisasi / underfit)
#   2) di antara sisanya, ambil F1 klasifikasi (threshold median) tertinggi
# Angka r2_test/accuracy/f1 di bawah adalah hasil ASLI dari
# `regresi_results` & `klasifikasi_results` pada notebook (train 2021-2024,
# test 2025), BUKAN simulasi.
BEST_MODEL_PER_TARGET = {
    'ARPS_07to12': {'model': 'XGBoost', 'r2_test': 0.848962, 'accuracy': 0.842105, 'f1': 0.842105,
                     'is_ensemble': True},
    'ARPS_13to15': {'model': 'Extra Trees', 'r2_test': 0.909494, 'accuracy': 0.868421, 'f1': 0.848485,
                     'is_ensemble': True},
    'ARPS_16to18': {'model': 'XGBoost', 'r2_test': 0.892577, 'accuracy': 0.894737, 'f1': 0.750000,
                     'is_ensemble': True},
    'ARPS_19to23': {'model': 'KNN Regressor', 'r2_test': 0.776713, 'accuracy': 0.894737, 'f1': 0.866667,
                     'is_ensemble': False},
}

SCALED_MODEL_NAMES = {'Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet', 'KNN Regressor'}


def _make_model(name: str):
    """Instansiasi model dengan hiperparameter PERSIS sama dengan pipeline
    regresi pada notebook (cell 14 — `run_pipeline`), karena angka r2_test
    yang dipakai untuk memilih model terbaik berasal dari pipeline ini."""
    if name == 'XGBoost':
        return XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
    if name == 'Extra Trees':
        return ExtraTreesRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    if name == 'Gradient Boosting':
        return GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    if name == 'KNN Regressor':
        return KNeighborsRegressor(n_neighbors=5)
    raise ValueError(f"Model tidak dikenal: {name}")


# ── Tabel hasil ASLI dari notebook (regresi & klasifikasi) ─────────────────
# Disalin persis dari output `regresi_results` / `klasifikasi_results` pada
# sahda_DropAlert_Final.ipynb (train 2021-2024, test 2025, n_test=38).

_REGRESI_ROWS = [
    ('ARPS_07to12', 'Gradient Boosting', 0.999990, 0.921003, 1.178378, 0.456443),
    ('ARPS_07to12', 'XGBoost', 0.999999, 0.848962, 1.629379, 0.592830),
    ('ARPS_07to12', 'Extra Trees', 0.999968, 0.491359, 2.990090, 0.767778),
    ('ARPS_07to12', 'Decision Tree', 0.988509, -0.910521, 5.795013, 1.625264),
    ('ARPS_07to12', 'Random Forest', 0.984525, -1.157764, 6.158578, 1.541117),
    ('ARPS_07to12', 'Linear Regression', 0.939033, -2.836342, 8.211777, 5.794631),
    ('ARPS_07to12', 'Lasso Regression', 0.934067, -2.909008, 8.289183, 5.856348),
    ('ARPS_07to12', 'Ridge Regression', 0.933245, -3.093836, 8.482888, 6.042737),
    ('ARPS_07to12', 'ElasticNet', 0.931944, -3.160292, 8.551462, 6.133225),
    ('ARPS_07to12', 'KNN Regressor', 0.981975, -3.501736, 8.895461, 2.191974),
    ('ARPS_13to15', 'Gradient Boosting', 0.999941, 0.975502, 0.982051, 0.724946),
    ('ARPS_13to15', 'XGBoost', 0.999996, 0.971669, 1.056094, 0.785414),
    ('ARPS_13to15', 'Extra Trees', 0.999420, 0.909494, 1.887603, 0.915577),
    ('ARPS_13to15', 'Random Forest', 0.987704, 0.491777, 4.473002, 1.549965),
    ('ARPS_13to15', 'Decision Tree', 0.985787, 0.308483, 5.217631, 2.460283),
    ('ARPS_13to15', 'Linear Regression', 0.947969, -0.129745, 6.669021, 5.002213),
    ('ARPS_13to15', 'Lasso Regression', 0.942938, -0.185994, 6.833025, 5.123094),
    ('ARPS_13to15', 'Ridge Regression', 0.942176, -0.209069, 6.899177, 5.187184),
    ('ARPS_13to15', 'ElasticNet', 0.940855, -0.227997, 6.952971, 5.245514),
    ('ARPS_13to15', 'KNN Regressor', 0.983571, -0.262396, 7.049682, 2.171605),
    ('ARPS_16to18', 'XGBoost', 0.999970, 0.892577, 2.894765, 2.287272),
    ('ARPS_16to18', 'Gradient Boosting', 0.999544, 0.887579, 2.961341, 2.388985),
    ('ARPS_16to18', 'Extra Trees', 0.996076, 0.855742, 3.354545, 2.694061),
    ('ARPS_16to18', 'Random Forest', 0.988101, 0.818017, 3.767721, 2.903671),
    ('ARPS_16to18', 'Decision Tree', 0.985681, 0.781718, 4.126411, 3.369437),
    ('ARPS_16to18', 'Lasso Regression', 0.934313, 0.607842, 5.530883, 4.378800),
    ('ARPS_16to18', 'Linear Regression', 0.935802, 0.593161, 5.633458, 4.422523),
    ('ARPS_16to18', 'ElasticNet', 0.933239, 0.592242, 5.639821, 4.496438),
    ('ARPS_16to18', 'Ridge Regression', 0.933986, 0.589355, 5.659748, 4.487559),
    ('ARPS_16to18', 'KNN Regressor', 0.978694, 0.545561, 5.953901, 3.820447),
    ('ARPS_19to23', 'XGBoost', 0.999901, 0.842911, 2.818889, 2.146823),
    ('ARPS_19to23', 'Gradient Boosting', 0.998656, 0.830598, 2.927278, 2.295814),
    ('ARPS_19to23', 'Extra Trees', 0.984709, 0.813204, 3.073895, 2.301429),
    ('ARPS_19to23', 'KNN Regressor', 0.962056, 0.776713, 3.360755, 2.261789),
    ('ARPS_19to23', 'Random Forest', 0.970569, 0.730139, 3.694664, 2.815266),
    ('ARPS_19to23', 'Linear Regression', 0.874926, 0.684800, 3.992985, 3.143340),
    ('ARPS_19to23', 'Ridge Regression', 0.873032, 0.681750, 4.012261, 3.202939),
    ('ARPS_19to23', 'Lasso Regression', 0.871466, 0.678129, 4.035022, 3.203890),
    ('ARPS_19to23', 'ElasticNet', 0.869748, 0.674718, 4.056344, 3.243413),
    ('ARPS_19to23', 'Decision Tree', 0.943755, 0.669253, 4.090276, 3.053136),
]
REGRESI_RESULTS = pd.DataFrame(
    _REGRESI_ROWS, columns=['target', 'model', 'r2_train', 'r2_test', 'rmse', 'mae']
)

_KLASIFIKASI_ROWS = [
    ('ARPS_07to12', 'KNN Regressor', 0.894737, 0.894737),
    ('ARPS_07to12', 'Extra Trees', 0.868421, 0.878049),
    ('ARPS_07to12', 'Decision Tree', 0.842105, 0.850000),
    ('ARPS_07to12', 'XGBoost', 0.842105, 0.842105),
    ('ARPS_07to12', 'Gradient Boosting', 0.815789, 0.820513),
    ('ARPS_07to12', 'Random Forest', 0.789474, 0.809524),
    ('ARPS_07to12', 'Linear Regression', 0.631579, 0.695652),
    ('ARPS_07to12', 'Ridge Regression', 0.578947, 0.652174),
    ('ARPS_07to12', 'Lasso Regression', 0.578947, 0.652174),
    ('ARPS_07to12', 'ElasticNet', 0.578947, 0.652174),
    ('ARPS_13to15', 'Extra Trees', 0.868421, 0.848485),
    ('ARPS_13to15', 'KNN Regressor', 0.842105, 0.823529),
    ('ARPS_13to15', 'XGBoost', 0.815789, 0.787879),
    ('ARPS_13to15', 'ElasticNet', 0.763158, 0.780488),
    ('ARPS_13to15', 'Random Forest', 0.815789, 0.774194),
    ('ARPS_13to15', 'Gradient Boosting', 0.815789, 0.774194),
    ('ARPS_13to15', 'Linear Regression', 0.736842, 0.761905),
    ('ARPS_13to15', 'Ridge Regression', 0.736842, 0.761905),
    ('ARPS_13to15', 'Lasso Regression', 0.736842, 0.761905),
    ('ARPS_13to15', 'Decision Tree', 0.763158, 0.666667),
    ('ARPS_16to18', 'XGBoost', 0.894737, 0.750000),
    ('ARPS_16to18', 'Gradient Boosting', 0.868421, 0.705882),
    ('ARPS_16to18', 'Random Forest', 0.815789, 0.631579),
    ('ARPS_16to18', 'Extra Trees', 0.842105, 0.625000),
    ('ARPS_16to18', 'Linear Regression', 0.789474, 0.555556),
    ('ARPS_16to18', 'Ridge Regression', 0.789474, 0.555556),
    ('ARPS_16to18', 'Lasso Regression', 0.789474, 0.555556),
    ('ARPS_16to18', 'ElasticNet', 0.789474, 0.555556),
    ('ARPS_16to18', 'Decision Tree', 0.710526, 0.476190),
    ('ARPS_16to18', 'KNN Regressor', 0.710526, 0.476190),
    ('ARPS_19to23', 'KNN Regressor', 0.894737, 0.866667),
    ('ARPS_19to23', 'XGBoost', 0.842105, 0.785714),
    ('ARPS_19to23', 'Random Forest', 0.815789, 0.774194),
    ('ARPS_19to23', 'Extra Trees', 0.815789, 0.758621),
    ('ARPS_19to23', 'Decision Tree', 0.789474, 0.750000),
    ('ARPS_19to23', 'Linear Regression', 0.815789, 0.740741),
    ('ARPS_19to23', 'Ridge Regression', 0.815789, 0.740741),
    ('ARPS_19to23', 'Lasso Regression', 0.815789, 0.740741),
    ('ARPS_19to23', 'ElasticNet', 0.815789, 0.740741),
    ('ARPS_19to23', 'Gradient Boosting', 0.815789, 0.740741),
]
KLASIFIKASI_RESULTS = pd.DataFrame(
    _KLASIFIKASI_ROWS, columns=['target', 'model', 'accuracy', 'f1']
)

# Korelasi fitur terhadap tiap target ARPS — angka asli dari notebook cell
# "KORELASI TERHADAP TARGET" (Pearson, data provinsi x tahun setelah IQR-capping).
CORR_TARGET = pd.DataFrame({
    'fitur': ['gabungan_pendudukmiskin', 'TPT', 'NEET_usiamuda', 'tenagakerjaformal',
              'gabungan_HLS', 'gabungan_RLS',
              'rasio_murid_guru_SMA', 'rasio_murid_guru_SMK', 'rasio_murid_guru_SD', 'rasio_murid_guru_SMP',
              'rasio_murid_sekolah_SMA', 'rasio_murid_sekolah_SMK', 'rasio_murid_sekolah_SD', 'rasio_murid_sekolah_SMP'],
    'ARPS_07to12': [-0.073439, -0.674793, -0.583500, -0.637383, -0.675647, -0.722251,
                     0.033768, -0.059057, 0.478343, 0.170666, -0.303717, -0.210567, 0.099113, -0.082122],
    'ARPS_13to15': [-0.057525, -0.704778, -0.560863, -0.666536, -0.708021, -0.761339,
                     0.048846, -0.065022, 0.478107, 0.166916, -0.297531, -0.227222, 0.087339, -0.107237],
    'ARPS_16to18': [-0.059091, -0.795134, -0.430419, -0.663788, -0.796466, -0.811281,
                     0.115098, -0.013636, 0.510087, 0.194445, -0.290542, -0.253171, 0.106344, -0.124049],
    'ARPS_19to23': [-0.196258, -0.841334, -0.268384, -0.433348, -0.839401, -0.678073,
                     0.239003, 0.168068, 0.512425, 0.292751, -0.204069, -0.168069, 0.265582, -0.029462],
})

# Silhouette score k=2..9 — angka asli dari notebook (cell 27). k=3 dipilih
# karena silhouette tertinggi (0.3092) di antara seluruh k yang diuji.
SILHOUETTE_SCORES = {2: 0.2563, 3: 0.3092, 4: 0.2545, 5: 0.2416,
                      6: 0.2540, 7: 0.2159, 8: 0.2561, 9: 0.2250}
OPTIMAL_K = 3

CLUSTER_CSV_PATH = "dropalert_cluster_provinsi.csv"

# Alias provinsi — DISESUAIKAN agar mengarah ke nama persis seperti pada
# dropalert_cluster_provinsi.csv / notebook (mis. "KEP. RIAU", bukan
# "KEPULAUAN RIAU" seperti versi lama, yang membuat pencocokan gagal).
PROVINSI_ALIAS = {
    'diy': 'DI YOGYAKARTA', 'yogyakarta': 'DI YOGYAKARTA', 'di yogyakarta': 'DI YOGYAKARTA',
    'daerah istimewa yogyakarta': 'DI YOGYAKARTA', 'jogja': 'DI YOGYAKARTA',
    'jakarta': 'DKI JAKARTA', 'dki jakarta': 'DKI JAKARTA', 'dki': 'DKI JAKARTA',
    'aceh': 'ACEH', 'nad': 'ACEH', 'nanggroe aceh darussalam': 'ACEH',
    'kalbar': 'KALIMANTAN BARAT', 'kalimantan barat': 'KALIMANTAN BARAT',
    'kaltim': 'KALIMANTAN TIMUR', 'kalimantan timur': 'KALIMANTAN TIMUR',
    'kaltara': 'KALIMANTAN UTARA', 'kalimantan utara': 'KALIMANTAN UTARA',
    'kalsel': 'KALIMANTAN SELATAN', 'kalimantan selatan': 'KALIMANTAN SELATAN',
    'kalteng': 'KALIMANTAN TENGAH', 'kalimantan tengah': 'KALIMANTAN TENGAH',
    'sulut': 'SULAWESI UTARA', 'sulawesi utara': 'SULAWESI UTARA',
    'sulsel': 'SULAWESI SELATAN', 'sulawesi selatan': 'SULAWESI SELATAN',
    'sulteng': 'SULAWESI TENGAH', 'sulawesi tengah': 'SULAWESI TENGAH',
    'sultra': 'SULAWESI TENGGARA', 'sulawesi tenggara': 'SULAWESI TENGGARA',
    'sulbar': 'SULAWESI BARAT', 'sulawesi barat': 'SULAWESI BARAT',
    'sumut': 'SUMATERA UTARA', 'sumatera utara': 'SUMATERA UTARA',
    'sumbar': 'SUMATERA BARAT', 'sumatera barat': 'SUMATERA BARAT',
    'sumsel': 'SUMATERA SELATAN', 'sumatera selatan': 'SUMATERA SELATAN',
    'riau': 'RIAU',
    'kepri': 'KEP. RIAU', 'kepulauan riau': 'KEP. RIAU', 'kep riau': 'KEP. RIAU', 'kep. riau': 'KEP. RIAU',
    'jambi': 'JAMBI', 'bengkulu': 'BENGKULU', 'lampung': 'LAMPUNG',
    'babel': 'KEP. BANGKA BELITUNG', 'bangka belitung': 'KEP. BANGKA BELITUNG',
    'kepulauan bangka belitung': 'KEP. BANGKA BELITUNG', 'kep. bangka belitung': 'KEP. BANGKA BELITUNG',
    'banten': 'BANTEN',
    'jabar': 'JAWA BARAT', 'jawa barat': 'JAWA BARAT',
    'jateng': 'JAWA TENGAH', 'jawa tengah': 'JAWA TENGAH',
    'jatim': 'JAWA TIMUR', 'jawa timur': 'JAWA TIMUR',
    'bali': 'BALI',
    'ntb': 'NUSA TENGGARA BARAT', 'nusa tenggara barat': 'NUSA TENGGARA BARAT',
    'ntt': 'NUSA TENGGARA TIMUR', 'nusa tenggara timur': 'NUSA TENGGARA TIMUR',
    'maluku': 'MALUKU', 'malut': 'MALUKU UTARA', 'maluku utara': 'MALUKU UTARA',
    'papua': 'PAPUA', 'papua barat': 'PAPUA BARAT', 'papua selatan': 'PAPUA SELATAN',
    'papua tengah': 'PAPUA TENGAH', 'papua pegunungan': 'PAPUA PEGUNUNGAN',
    'papua barat daya': 'PAPUA BARAT DAYA', 'gorontalo': 'GORONTALO',
}

RAW_DATASET_CANDIDATES = [
    "Dataset_Gabungan_Fix.xlsx",
    "Dataset_Gabungan.xlsx",
    "data/Dataset_Gabungan_Fix.xlsx",
    "data/Dataset_Gabungan.xlsx",
]

RAW_REQUIRED_COLUMNS = [
    'Provinsi', 'Tahun', 'kota_pendudukmiskin', 'desa_pendudukmiskin', 'gabungan_pendudukmiskin',
    'NEET_usiamuda', 'tenagakerjaformal',
    'SMA_gabungan_jumlahsekolah', 'SMA_gabungan_jumlahguru', 'SMA_gabungan_jumlahmurid',
    'SMK_gabungan_jumlahsekolah', 'SMK_gabungan_jumlahguru', 'SMK_gabungan_jumlahmurid',
    'SD_gabungan_jumlahsekolah', 'SD_gabungan_jumlahguru', 'SD_gabungan_jumlahmurid',
    'SMP_gabungan_jumlahsekolah', 'SMP_gabungan_jumlahguru', 'SMP_gabungan_jumlahmurid',
    'gabungan_RLS', 'gabungan_HLS', 'TPT',
    'APS_07to12', 'APS_13to15', 'APS_16to18', 'APS_19to23',
]


def mapping_input_provinsi(raw: str, valid_list: list):
    """Memetakan input provinsi bebas (alias/singkatan) ke nama baku pada
    dataset (dipakai di kotak pencarian provinsi pada halaman Prediksi)."""
    raw_clean = raw.strip().lower()
    if raw_clean in PROVINSI_ALIAS:
        candidate = PROVINSI_ALIAS[raw_clean]
        return candidate if candidate in valid_list else None
    for prov in valid_list:
        if prov.lower() == raw_clean:
            return prov
    for prov in valid_list:
        if raw_clean in prov.lower() or prov.lower() in raw_clean:
            return prov
    return None


# ════════════════════════════════════════════════════════
# 4. LOAD DATA — KLASTER (CSV, single source of truth)
# ════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_cluster_data(path: str = CLUSTER_CSV_PATH):
    """Memuat dropalert_cluster_provinsi.csv APA ADANYA — cluster, PC1, PC2,
    dan ke-14 fitur klastering dipakai langsung dari file ini, TANPA
    dihitung ulang di dashboard (beda dengan versi lama yang melakukan
    KMeans ulang per-tahun dengan subset fitur yang berbeda dari notebook)."""
    if not os.path.exists(path):
        return None
    df_cl = pd.read_csv(path)
    df_cl = df_cl.dropna(subset=['Provinsi']).copy()
    df_cl['Provinsi'] = df_cl['Provinsi'].astype(str).str.upper().str.strip()
    df_cl['cluster'] = df_cl['cluster'].astype(int)
    return df_cl


def compute_cluster_risk_label(df_cl: pd.DataFrame) -> dict:
    """Memberi label deskriptif (Rendah/Sedang/Tinggi) per cluster berdasarkan
    indeks komposit dari 5 indikator yang punya arah korelasi paling konsisten
    & paling mudah diinterpretasikan terhadap ARPS pada notebook (kemiskinan,
    NEET, tenaga kerja formal, HLS, RLS). TPT sengaja TIDAK dipakai di indeks
    ini karena arah korelasinya terhadap ARPS pada notebook counter-intuitive
    (provinsi Papua justru mencatat TPT terendah meski kondisi pendidikannya
    paling tertekan — TPT di sini lebih mencerminkan tingkat formalitas pasar
    kerja daripada tekanan ekonomi riil). Label ini murni bantuan naratif;
    identitas cluster asli (0/1/2, sesuai notebook & CSV) selalu ditampilkan
    berdampingan agar tidak menggantikan hasil KMeans yang sebenarnya."""
    cols = ['gabungan_pendudukmiskin', 'NEET_usiamuda', 'tenagakerjaformal', 'gabungan_HLS', 'gabungan_RLS']
    z = df_cl[cols].apply(lambda s: (s - s.mean()) / s.std(ddof=0))
    risk_index = z['gabungan_pendudukmiskin'] + z['NEET_usiamuda'] - z['tenagakerjaformal'] - z['gabungan_HLS'] - z['gabungan_RLS']
    tmp = df_cl.copy()
    tmp['_risk_index'] = risk_index
    ordered = tmp.groupby('cluster')['_risk_index'].mean().sort_values()
    labels = ['Rendah', 'Sedang', 'Tinggi']
    # Jika suatu saat jumlah cluster != 3 (mis. notebook diubah), tetap aman:
    n = len(ordered)
    if n <= 3:
        chosen_labels = labels[:n]
    else:
        chosen_labels = [f"Tingkat {i+1}" for i in range(n)]
    return {cl: lab for cl, lab in zip(ordered.index, chosen_labels)}


@st.cache_data(show_spinner=False)
def load_geojson():
    path = "38 Provinsi Indonesia - Provinsi.json"
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        geo = json.load(file)
        for feature in geo['features']:
            if 'PROVINSI' in feature['properties']:
                feature['properties']['PROVINSI'] = str(feature['properties']['PROVINSI']).upper()
        return geo


# ════════════════════════════════════════════════════════
# 5. LOAD DATA MENTAH (OPSIONAL) — untuk fitur Prediksi & EDA per-tahun
# ════════════════════════════════════════════════════════
#
# Dashboard TIDAK menyertakan dataset mentah (Dataset_Gabungan_Fix.xlsx)
# karena tidak diunggah bersama notebook/CSV. Fitur yang butuh data
# panel per-tahun (prediksi interaktif, tren ARPS, EDA distribusi mentah)
# baru aktif jika file ini ditemukan di folder yang sama dengan app.py,
# atau diunggah manual lewat sidebar. Tanpa file ini, dashboard tetap bisa
# menampilkan seluruh hasil klastering (CSV) dan seluruh tabel evaluasi
# model (angka asli notebook, sudah hardcoded di atas) — supaya dashboard
# tidak pernah "kosong", tapi juga tidak pernah menampilkan prediksi dari
# model abal-abal.

def find_local_raw_dataset():
    for p in RAW_DATASET_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


@st.cache_data(show_spinner=False)
def load_raw_panel(file_or_path) -> pd.DataFrame:
    """Load dataset mentah (path lokal atau file upload) dan bersihkan
    persis seperti awal notebook: hapus baris 'INDONESIA', kolom Provinsi
    di-uppercase."""
    df = pd.read_excel(file_or_path)
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Kolom wajib berikut tidak ditemukan pada dataset yang diunggah: " + ", ".join(missing)
        )
    df['Provinsi'] = df['Provinsi'].astype(str).str.upper().str.strip()
    df = df[df['Provinsi'] != 'INDONESIA'].copy()
    return df


def cap_outliers_iqr(data: pd.DataFrame, columns: list, factor: float = 1.5):
    """Replikasi PERSIS fungsi `cap_outliers_iqr` pada notebook (cell 9):
    Winsorization berbasis IQR, dihitung dari seluruh baris yang diberikan
    (bukan hanya train). Dashboard sengaja mereplikasi perilaku ini apa
    adanya agar angka yang dihasilkan konsisten dengan notebook — lihat
    catatan metodologis di halaman Evaluasi Model soal keterbatasan
    pendekatan ini (batas capping dihitung dari gabungan train+test)."""
    data = data.copy()
    for col in columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        data[col] = data[col].clip(lower=lower, upper=upper)
    return data


def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering PERSIS sama dengan notebook cell 4: ARPS = 100 -
    APS, rasio murid-per-guru & murid-per-sekolah (BUKAN kebalikannya),
    lalu IQR-capping pada kolom fitur X (bukan target)."""
    df = df_raw.copy()

    df['ARPS_07to12'] = 100 - df['APS_07to12']
    df['ARPS_13to15'] = 100 - df['APS_13to15']
    df['ARPS_16to18'] = 100 - df['APS_16to18']
    df['ARPS_19to23'] = 100 - df['APS_19to23']

    df['rasio_murid_guru_SMA'] = df['SMA_gabungan_jumlahmurid'] / df['SMA_gabungan_jumlahguru']
    df['rasio_murid_guru_SMK'] = df['SMK_gabungan_jumlahmurid'] / df['SMK_gabungan_jumlahguru']
    df['rasio_murid_guru_SD'] = df['SD_gabungan_jumlahmurid'] / df['SD_gabungan_jumlahguru']
    df['rasio_murid_guru_SMP'] = df['SMP_gabungan_jumlahmurid'] / df['SMP_gabungan_jumlahguru']

    df['rasio_murid_sekolah_SMA'] = df['SMA_gabungan_jumlahmurid'] / df['SMA_gabungan_jumlahsekolah']
    df['rasio_murid_sekolah_SMK'] = df['SMK_gabungan_jumlahmurid'] / df['SMK_gabungan_jumlahsekolah']
    df['rasio_murid_sekolah_SD'] = df['SD_gabungan_jumlahmurid'] / df['SD_gabungan_jumlahsekolah']
    df['rasio_murid_sekolah_SMP'] = df['SMP_gabungan_jumlahmurid'] / df['SMP_gabungan_jumlahsekolah']

    keep = ['Tahun', 'Provinsi'] + FITUR + TARGET_LIST
    df_model = df[keep].copy()
    df_model = cap_outliers_iqr(df_model, FITUR)
    return df_model


# ════════════════════════════════════════════════════════
# 6. TRAINING MODEL — hiperparameter & split waktu PERSIS notebook
# ════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def train_all_target_models(df_model: pd.DataFrame, train_end_year: int = 2024) -> dict:
    """Melatih, untuk tiap target ARPS, PERSIS model yang dipilih notebook
    sebagai 'model terbaik' (BEST_MODEL_PER_TARGET) dengan split waktu yang
    sama (train <= train_end_year, test > train_end_year) dan hiperparameter
    yang sama dengan pipeline regresi notebook (cell 14). Threshold
    klasifikasi memakai median target dari data TRAIN saja (persis notebook
    cell 19 — "threshold selalu dari y_train, jangan dihitung ulang dari
    gabungan train+test").
    """
    df_sorted = df_model.sort_values(['Tahun', 'Provinsi'])
    train = df_sorted[df_sorted['Tahun'] <= train_end_year]
    test = df_sorted[df_sorted['Tahun'] > train_end_year]

    X_train_raw = train[FITUR]
    X_test_raw = test[FITUR] if len(test) else None

    trained = {}
    for target in TARGET_LIST:
        cfg = BEST_MODEL_PER_TARGET[target]
        model_name = cfg['model']

        y_train = train[target]
        y_test = test[target] if len(test) else None

        scaler = None
        if model_name in SCALED_MODEL_NAMES:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw) if X_test_raw is not None else None
        else:
            X_train = X_train_raw.values
            X_test = X_test_raw.values if X_test_raw is not None else None

        model = _make_model(model_name)
        model.fit(X_train, y_train)

        r2_test_live = None
        if X_test is not None and len(y_test) > 0:
            r2_test_live = float(r2_score(y_test, model.predict(X_test)))

        trained[target] = {
            'model': model,
            'scaler': scaler,
            'model_name': model_name,
            'train_median': float(y_train.median()),
            'r2_test_live': r2_test_live,
            'n_train': len(train),
            'n_test': len(test),
        }

    return trained


def predict_arps(trained: dict, target: str, X_row: pd.Series) -> dict:
    """Prediksi ARPS + klasifikasi biner (persis notebook: >= median train =
    kelas 1 / berisiko lebih tinggi dari median historis)."""
    cfg = trained[target]
    X_vec = X_row[FITUR].values.reshape(1, -1).astype(float)
    if cfg['scaler'] is not None:
        X_vec = cfg['scaler'].transform(X_vec)
    arps_pred = float(cfg['model'].predict(X_vec)[0])
    arps_pred = float(np.clip(arps_pred, 0, 100))
    aps_pred = 100 - arps_pred

    median = cfg['train_median']
    above_median = arps_pred >= median

    # Lapisan deskriptif 4-tingkat (UNTUK VISUALISASI SAJA, bukan metrik yang
    # dievaluasi di notebook — notebook hanya menguji klasifikasi BINER
    # terhadap median). Kelipatan median di bawah ini murni bantuan UX.
    if arps_pred >= median * 1.25:
        risk_level, risk_label, color = 'tinggi', 'TINGGI — Prioritas Intervensi', '#C0392B'
    elif arps_pred >= median:
        risk_level, risk_label, color = 'sedang-tinggi', 'DI ATAS MEDIAN — Perlu Perhatian', '#E67E22'
    elif arps_pred >= median * 0.75:
        risk_level, risk_label, color = 'sedang-rendah', 'DI BAWAH MEDIAN — Pemantauan Rutin', '#F39C12'
    else:
        risk_level, risk_label, color = 'rendah', 'RENDAH — Kondisi Relatif Stabil', '#27AE60'

    return {
        'arps': arps_pred,
        'aps': aps_pred,
        'median': median,
        'above_median': bool(above_median),
        'risk_level': risk_level,
        'risk_label': risk_label,
        'color': color,
        'model_name': cfg['model_name'],
    }


@st.cache_data(show_spinner=False)
def compute_permutation_importance(_trained_entry: dict, df_model: pd.DataFrame, target: str, train_end_year: int = 2024):
    """Feature importance GENERIK via permutation importance (bekerja untuk
    semua jenis model, termasuk KNN yang tidak punya `.feature_importances_`)
    dihitung pada data test (2025) — menggantikan chart feature-importance
    lama yang angkanya disimulasikan/tidak berasal dari model manapun."""
    df_sorted = df_model.sort_values(['Tahun', 'Provinsi'])
    test = df_sorted[df_sorted['Tahun'] > train_end_year]
    if len(test) < 5:
        return None

    cfg = _trained_entry
    X_test_raw = test[FITUR]
    y_test = test[target]
    X_test = cfg['scaler'].transform(X_test_raw) if cfg['scaler'] is not None else X_test_raw.values

    result = permutation_importance(
        cfg['model'], X_test, y_test, n_repeats=20, random_state=42, scoring='r2'
    )
    imp_df = pd.DataFrame({
        'fitur': FITUR,
        'importance': result.importances_mean,
    }).sort_values('importance', ascending=False)
    return imp_df


# ════════════════════════════════════════════════════════
# 7. BUILD DASHBOARD
# ════════════════════════════════════════════════════════

def build_dashboard():
    with st.sidebar:
        st.markdown("<h1 style='color:#E74C3C; margin-bottom:2px;'>🚨 DropAlert</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#AAA; margin-top:0; font-size:.85rem;'>Deteksi Risiko Putus Sekolah & Pemetaan Klaster Provinsi</p>", unsafe_allow_html=True)
        st.markdown("---")
        menu = st.radio(
            "Navigasi",
            ["Beranda", "Peta & Klaster Provinsi", "Prediksi Risiko",
             "Evaluasi Model", "EDA & Korelasi", "Tentang Proyek"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("<p style='color:#888; font-size:0.75rem;'>Dataset panel per-tahun (opsional)</p>", unsafe_allow_html=True)
        uploaded_raw = st.file_uploader(
            "Unggah Dataset_Gabungan_Fix.xlsx untuk mengaktifkan prediksi & EDA per-tahun",
            type=['xlsx'], label_visibility="collapsed",
        )
        st.caption(
            "Tanpa file ini, dashboard tetap menampilkan seluruh hasil klastering "
            "(CSV) & tabel evaluasi model asli notebook — hanya prediksi interaktif "
            "dan EDA per-tahun yang nonaktif."
        )
        st.markdown("---")
        st.markdown("<div style='text-align: center; color: #888; font-size: 0.75rem; margin-top: 1rem;'>© DROPALERT 2026<br>IT FEST 6.0</div>", unsafe_allow_html=True)

    # ── Load data klaster (wajib) ───────────────────────
    df_cl = load_cluster_data()
    if df_cl is None:
        st.error(
            f"⚠️ **File `{CLUSTER_CSV_PATH}` tidak ditemukan.**\n\n"
            "Letakkan file ini pada folder yang sama dengan `app.py`. File ini "
            "wajib ada karena menjadi satu-satunya sumber hasil klastering "
            "provinsi yang ditampilkan di dashboard."
        )
        return
    cluster_risk_map = compute_cluster_risk_label(df_cl)
    df_cl = df_cl.copy()
    df_cl['risk_label'] = df_cl['cluster'].map(cluster_risk_map)

    # ── Load data mentah (opsional) ─────────────────────
    raw_source = uploaded_raw if uploaded_raw is not None else find_local_raw_dataset()
    df_model = None
    trained = None
    raw_error = None
    if raw_source is not None:
        try:
            df_raw = load_raw_panel(raw_source)
            df_model = engineer_features(df_raw)
            trained = train_all_target_models(df_model)
        except Exception as e:
            raw_error = str(e)

    # ════════════════════════════════════════════════════
    # A. BERANDA
    # ════════════════════════════════════════════════════
    if menu == "Beranda":
        st.markdown("""
        <div class="hero-container">
          <h1>DropAlert</h1>
          <h1>Deteksi Risiko Putus Sekolah &amp; Pemetaan Klaster Provinsi<br>Berbasis Dashboard Interaktif</h1>
          <p>Membandingkan berbagai algoritma regresi (linear, tree-based, ensemble, KNN) dan
          K-Means clustering untuk mendukung intervensi pendidikan presisi &middot; Sumber Data BPS 2021&ndash;2025</p>
        </div>
        """, unsafe_allow_html=True)

        best_overall = REGRESI_RESULTS.loc[REGRESI_RESULTS['r2_test'].idxmax()]
        c1, c2, c3, c4 = st.columns(4)
        kpis = [
            ("Provinsi Dianalisis", f"{df_cl['Provinsi'].nunique()}", "rata-rata indikator 2021–2025"),
            ("Jumlah Klaster (K-Means)", f"{OPTIMAL_K}", f"silhouette = {SILHOUETTE_SCORES[OPTIMAL_K]:.3f}"),
            ("R² Test Tertinggi", f"{best_overall['r2_test']:.3f}", f"{best_overall['model']} · {best_overall['target']}"),
            ("Target ARPS Dimodelkan", "4", "usia 7–12, 13–15, 16–18, 19–23"),
        ]
        for col, (lbl, val, sub) in zip([c1, c2, c3, c4], kpis):
            col.markdown(
                f'<div class="kpi-card"><div class="label">{lbl}</div>'
                f'<div class="value">{val}</div><div class="sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div class='section-title'>Peta Klaster Provinsi (berdasarkan rata-rata indikator 2021–2025)</div>", unsafe_allow_html=True)
        st.markdown(
            "Peta ini memuat langsung hasil K-Means dari `dropalert_cluster_provinsi.csv` "
            "(14 fitur, k=3) — klaster bersifat **level-provinsi** (rata-rata lima tahun), "
            "bukan per-tahun, sehingga keanggotaan klaster tidak berubah saat memfilter tahun."
        )

        indo_geojson = load_geojson()
        if indo_geojson:
            fig_map = px.choropleth_mapbox(
                df_cl,
                geojson=indo_geojson,
                locations="Provinsi",
                featureidkey="properties.PROVINSI",
                color="risk_label",
                category_orders={"risk_label": ["Rendah", "Sedang", "Tinggi"]},
                color_discrete_map={"Tinggi": "#C0392B", "Sedang": "#F39C12", "Rendah": "#27AE60"},
                mapbox_style="carto-darkmatter",
                zoom=3.3,
                center={"lat": -2.5, "lon": 118.0},
                opacity=0.85,
                hover_name="Provinsi",
                hover_data={"cluster": True, "gabungan_pendudukmiskin": ':.2f', "gabungan_RLS": ':.2f', "risk_label": False},
            )
            fig_map.update_layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="Profil Klaster",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("File GeoJSON `38 Provinsi Indonesia - Provinsi.json` tidak ditemukan — peta koropleth dilewati, lihat tabel & scatter PCA di bawah.")

        lc1, lc2, lc3 = st.columns(3)
        cluster_desc = {
            'Rendah': "Kemiskinan & NEET relatif rendah, tenaga kerja formal & lama sekolah relatif tinggi.",
            'Sedang': "Profil sosial-ekonomi menengah — perlu pemantauan berkala.",
            'Tinggi': "RLS/HLS & tenaga kerja formal paling rendah di antara ketiga klaster — didominasi wilayah Papua.",
        }
        for col, lab, color in zip([lc1, lc2, lc3], ["Rendah", "Sedang", "Tinggi"], ["#27AE60", "#F39C12", "#C0392B"]):
            n_prov = int((df_cl['risk_label'] == lab).sum())
            col.markdown(
                f"""<div class="insight-box" style="border-left-color:{color};">
                <strong>Klaster {lab}</strong> ({n_prov} provinsi)<br><small>{cluster_desc[lab]}</small></div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<div class='section-title'>Visualisasi PCA (2 Komponen)</div>", unsafe_allow_html=True)
        st.caption("PC1/PC2 diambil langsung dari CSV (hasil PCA notebook atas 14 fitur klastering yang sudah distandardisasi).")
        fig_pca = px.scatter(
            df_cl, x='PC1', y='PC2', color='risk_label', hover_name='Provinsi',
            category_orders={"risk_label": ["Rendah", "Sedang", "Tinggi"]},
            color_discrete_map={"Tinggi": "#C0392B", "Sedang": "#F39C12", "Rendah": "#27AE60"},
            template='plotly_dark',
        )
        fig_pca.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=420, legend_title_text="Profil Klaster",
        )
        st.plotly_chart(fig_pca, use_container_width=True)

        if df_model is not None:
            st.markdown("<div class='section-title'>Tren ARPS Nasional per Kelompok Umur</div>", unsafe_allow_html=True)
            tren = df_model.groupby('Tahun')[TARGET_LIST].mean().reset_index()
            fig_tren = px.line(
                tren, x='Tahun', y=TARGET_LIST, template='plotly_dark', markers=True,
                color_discrete_sequence=['#3498DB', '#2ECC71', '#E74C3C', '#F39C12'],
                labels={'value': 'ARPS (%)', 'variable': 'Target'},
            )
            fig_tren.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=360)
            st.plotly_chart(fig_tren, use_container_width=True)
        else:
            st.info(
                "📈 Tren ARPS per tahun & Top-5 provinsi paling berisiko butuh dataset panel "
                "mentah (`Dataset_Gabungan_Fix.xlsx`). Unggah lewat sidebar untuk mengaktifkan."
            )
            if raw_error:
                st.warning(f"Dataset yang diunggah gagal diproses: {raw_error}")

    # ════════════════════════════════════════════════════
    # B. PETA & KLASTER PROVINSI (detail)
    # ════════════════════════════════════════════════════
    elif menu == "Peta & Klaster Provinsi":
        st.markdown("<h2 class='section-title'>Detail Klastering Provinsi (K-Means, k=3)</h2>", unsafe_allow_html=True)
        st.markdown(
            "Klaster dihitung di notebook menggunakan **14 fitur** yang sama dengan model "
            "prediksi (kemiskinan, TPT, NEET, tenaga kerja formal, HLS, RLS, serta 8 rasio "
            "murid-guru & murid-sekolah), distandardisasi (`StandardScaler`), dirata-ratakan "
            "per provinsi selama 2021–2025, lalu di-cluster dengan K-Means. Nilai **k=3** dipilih "
            "karena silhouette score tertinggi di antara k=2..9 yang diuji notebook."
        )

        with st.expander("Kurva Silhouette Score (k = 2..9) — angka asli notebook"):
            sil_df = pd.DataFrame({'k': list(SILHOUETTE_SCORES.keys()), 'silhouette': list(SILHOUETTE_SCORES.values())})
            fig_sil = px.line(sil_df, x='k', y='silhouette', markers=True, template='plotly_dark')
            fig_sil.add_vline(x=OPTIMAL_K, line_dash='dash', line_color='#F39C12')
            fig_sil.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=320)
            st.plotly_chart(fig_sil, use_container_width=True)

        st.markdown("<div class='section-title'>Profil Rata-rata Fitur per Klaster</div>", unsafe_allow_html=True)
        profile = df_cl.groupby(['cluster'])[CLUSTER_FEATURES].mean().round(3)
        profile.insert(0, 'Label Deskriptif', [cluster_risk_map[c] for c in profile.index])
        profile.insert(1, 'Jumlah Provinsi', df_cl.groupby('cluster').size().values)
        st.dataframe(profile, use_container_width=True)

        st.markdown("<div class='section-title'>Daftar Provinsi per Klaster</div>", unsafe_allow_html=True)
        sel_cluster_label = st.radio(
            "Pilih klaster:", options=["Rendah", "Sedang", "Tinggi"], horizontal=True, key="cluster_detail_radio"
        )
        subset = df_cl[df_cl['risk_label'] == sel_cluster_label].sort_values('Provinsi')
        cl1, cl2 = st.columns([1, 1.4])
        with cl1:
            cluster_id = subset['cluster'].iloc[0] if len(subset) else '-'
            st.markdown(
                f"""<div class="kpi-card">
                <div class="label">Klaster {sel_cluster_label} (id notebook: {cluster_id})</div>
                <div class="value">{len(subset)} Provinsi</div></div>""",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            for p in subset['Provinsi'].tolist():
                st.markdown(f"- {p}")
        with cl2:
            fig_scat = px.scatter(
                df_cl, x='gabungan_pendudukmiskin', y='gabungan_RLS', color='risk_label',
                category_orders={"risk_label": ["Rendah", "Sedang", "Tinggi"]},
                color_discrete_map={"Tinggi": "#C0392B", "Sedang": "#F39C12", "Rendah": "#27AE60"},
                hover_name='Provinsi', template='plotly_dark',
                labels={'gabungan_pendudukmiskin': 'Kemiskinan (%)', 'gabungan_RLS': 'Rata-rata Lama Sekolah (Tahun)'},
            )
            fig_scat.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=420, legend_title_text="Profil Klaster",
            )
            st.plotly_chart(fig_scat, use_container_width=True)

        st.markdown("<div class='section-title'>Data Klastering Lengkap</div>", unsafe_allow_html=True)
        st.dataframe(df_cl.drop(columns=['risk_label']).sort_values('cluster'), use_container_width=True, hide_index=True)
        st.caption("Kolom `cluster`, `PC1`, `PC2` diambil apa adanya dari dropalert_cluster_provinsi.csv.")

    # ════════════════════════════════════════════════════
    # C. PREDIKSI RISIKO
    # ════════════════════════════════════════════════════
    elif menu == "Prediksi Risiko":
        st.markdown("<h2 class='section-title'>Prediksi Risiko Putus Sekolah (ARPS)</h2>", unsafe_allow_html=True)

        if df_model is None:
            st.warning(
                "⚠️ **Fitur ini butuh dataset panel mentah.** Unggah "
                "`Dataset_Gabungan_Fix.xlsx` (kolom sama seperti pada notebook) lewat "
                "sidebar untuk mengaktifkan prediksi interaktif — model akan dilatih "
                "ulang persis mengikuti pipeline notebook (fitur, capping outlier, "
                "split waktu 2021–2024 vs 2025, dan model+hiperparameter yang sama "
                "dengan yang dipilih sebagai 'terbaik' di notebook)."
            )
            if raw_error:
                st.error(f"Dataset yang diunggah gagal diproses: {raw_error}")
            st.markdown("<div class='section-title'>Model yang akan dipakai per target</div>", unsafe_allow_html=True)
            info_tbl = pd.DataFrame([
                {'Target': t, 'Model Terbaik (notebook)': v['model'],
                 'R² Test': f"{v['r2_test']:.3f}", 'Accuracy': f"{v['accuracy']:.3f}", 'F1': f"{v['f1']:.3f}",
                 'Ensemble?': 'Ya' if v['is_ensemble'] else 'Tidak'}
                for t, v in BEST_MODEL_PER_TARGET.items()
            ])
            st.dataframe(info_tbl, use_container_width=True, hide_index=True)
            return

        st.markdown(
            "Pilih provinsi untuk mengambil data indikator terbaru sebagai titik awal, "
            "lalu sesuaikan nilainya jika perlu (mis. simulasi skenario kebijakan)."
        )
        provinsi_list = sorted(df_model['Provinsi'].unique())
        quick_search = st.text_input(
            "Cari cepat (opsional — terima singkatan/alias, mis. 'kepri', 'jogja', 'ntt'):", ""
        )
        default_idx = 0
        if quick_search.strip():
            matched = mapping_input_provinsi(quick_search, provinsi_list)
            if matched is not None:
                default_idx = provinsi_list.index(matched)
            else:
                st.caption(f"Tidak menemukan provinsi yang cocok dengan '{quick_search}'.")
        data_mode = st.selectbox("Mulai dari data provinsi:", provinsi_list, index=default_idx)
        prov_row = df_model[df_model['Provinsi'] == data_mode].sort_values('Tahun').iloc[-1]

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("#### Kondisi Sosial-Ekonomi")
            val_miskin = st.number_input("Persentase Penduduk Miskin (%)", 0.0, 100.0, float(prov_row['gabungan_pendudukmiskin']), 0.01, format="%.2f")
            val_tpt = st.number_input("Tingkat Pengangguran Terbuka / TPT (%)", 0.0, 100.0, float(prov_row['TPT']), 0.01, format="%.2f")
            val_neet = st.number_input("NEET Usia Muda 15–24 Tahun (%)", 0.0, 100.0, float(prov_row['NEET_usiamuda']), 0.01, format="%.2f")
            val_formal = st.number_input("Tenaga Kerja Formal (%)", 0.0, 100.0, float(prov_row['tenagakerjaformal']), 0.01, format="%.2f")
        with col_b:
            st.markdown("#### Capaian Pendidikan")
            val_hls = st.number_input("Harapan Lama Sekolah (Tahun)", 0.0, 20.0, float(prov_row['gabungan_HLS']), 0.01, format="%.2f")
            val_rls = st.number_input("Rata-rata Lama Sekolah (Tahun)", 0.0, 20.0, float(prov_row['gabungan_RLS']), 0.01, format="%.2f")
            val_rmg_sma = st.number_input("Rasio Murid/Guru SMA", 0.0, 100.0, float(prov_row['rasio_murid_guru_SMA']), 0.01, format="%.2f")
            val_rmg_smk = st.number_input("Rasio Murid/Guru SMK", 0.0, 100.0, float(prov_row['rasio_murid_guru_SMK']), 0.01, format="%.2f")
        with col_c:
            st.markdown("#### Infrastruktur Pendidikan")
            val_rmg_sd = st.number_input("Rasio Murid/Guru SD", 0.0, 100.0, float(prov_row['rasio_murid_guru_SD']), 0.01, format="%.2f")
            val_rmg_smp = st.number_input("Rasio Murid/Guru SMP", 0.0, 100.0, float(prov_row['rasio_murid_guru_SMP']), 0.01, format="%.2f")
            val_rms_sma = st.number_input("Rasio Murid/Sekolah SMA", 0.0, 2000.0, float(prov_row['rasio_murid_sekolah_SMA']), 0.1, format="%.1f")
            val_rms_smk = st.number_input("Rasio Murid/Sekolah SMK", 0.0, 2000.0, float(prov_row['rasio_murid_sekolah_SMK']), 0.1, format="%.1f")

        val_rms_sd = float(prov_row['rasio_murid_sekolah_SD'])
        val_rms_smp = float(prov_row['rasio_murid_sekolah_SMP'])
        with st.expander("Rasio Murid/Sekolah SD & SMP (memakai data provinsi terpilih)"):
            val_rms_sd = st.number_input("Rasio Murid/Sekolah SD", 0.0, 2000.0, val_rms_sd, 0.1, format="%.1f")
            val_rms_smp = st.number_input("Rasio Murid/Sekolah SMP", 0.0, 2000.0, val_rms_smp, 0.1, format="%.1f")

        if st.button("🚀 Jalankan Prediksi", use_container_width=True):
            X_input = pd.Series({
                'gabungan_pendudukmiskin': val_miskin, 'TPT': val_tpt, 'NEET_usiamuda': val_neet,
                'tenagakerjaformal': val_formal, 'gabungan_HLS': val_hls, 'gabungan_RLS': val_rls,
                'rasio_murid_guru_SMA': val_rmg_sma, 'rasio_murid_guru_SMK': val_rmg_smk,
                'rasio_murid_guru_SD': val_rmg_sd, 'rasio_murid_guru_SMP': val_rmg_smp,
                'rasio_murid_sekolah_SMA': val_rms_sma, 'rasio_murid_sekolah_SMK': val_rms_smk,
                'rasio_murid_sekolah_SD': val_rms_sd, 'rasio_murid_sekolah_SMP': val_rms_smp,
            })

            predictions = {t: predict_arps(trained, t, X_input) for t in TARGET_LIST}

            st.markdown("<br><h3 class='section-title'>Hasil Prediksi ARPS</h3>", unsafe_allow_html=True)
            gcols = st.columns(4)
            for col, target in zip(gcols, TARGET_LIST):
                res = predictions[target]
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=round(res['arps'], 2),
                    delta={'reference': res['median'], 'increasing': {'color': '#E74C3C'}, 'decreasing': {'color': '#27AE60'}},
                    number={'suffix': '%', 'font': {'size': 26, 'color': '#FFFFFF'}},
                    gauge={
                        'axis': {'range': [0, max(20, res['median'] * 2)], 'tickcolor': '#AAA', 'tickfont': {'size': 10}},
                        'bar': {'color': res['color']},
                        'bgcolor': '#1A1C23', 'borderwidth': 1, 'bordercolor': '#333',
                        'threshold': {'line': {'color': '#FFD700', 'width': 2}, 'thickness': 0.75, 'value': res['median']},
                    },
                ))
                fig_gauge.update_layout(height=210, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': '#FFFFFF', 'size': 11})
                with col:
                    st.markdown(f"<p style='text-align:center; color:#AAA; font-size:0.75rem;'>{TARGET_LABELS[target]}<br>model: {res['model_name']}</p>", unsafe_allow_html=True)
                    st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("<br><h3 class='section-title'>Ringkasan Prediksi</h3>", unsafe_allow_html=True)
            summary_rows = []
            for target in TARGET_LIST:
                res = predictions[target]
                summary_rows.append({
                    'Kelompok Usia': TARGET_LABELS[target],
                    'Model': res['model_name'],
                    'Prediksi ARPS (%)': f"{res['arps']:.3f}",
                    'Median Train (%)': f"{res['median']:.3f}",
                    'Klasifikasi biner notebook': 'DI ATAS median' if res['above_median'] else 'DI BAWAH median',
                    'Deskripsi 4-tingkat*': res['risk_label'],
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            st.caption(
                "*Kolom 'Deskripsi 4-tingkat' murni bantuan visual (kelipatan median), bukan "
                "metrik yang dievaluasi di notebook. Metrik yang divalidasi (accuracy/F1) hanya "
                "untuk klasifikasi BINER di atas/di bawah median — lihat halaman Evaluasi Model."
            )

            st.markdown("<br><h3 class='section-title'>Catatan Interpretasi</h3>", unsafe_allow_html=True)
            interp_lines = []
            for t in TARGET_LIST:
                strongest_t = (
                    CORR_TARGET[['fitur', t]]
                    .assign(abs_corr=lambda d: d[t].abs())
                    .sort_values('abs_corr', ascending=False)
                    .head(3)['fitur'].tolist()
                )
                interp_lines.append(f"<li><b>{TARGET_LABELS[t]}</b>: fitur berkorelasi terkuat — {', '.join(strongest_t)}</li>")
            st.markdown(f"""
            <div class="insight-box">
            Berdasarkan matriks korelasi notebook, fitur dengan hubungan paling kuat terhadap
            tiap target:
            <ul>{''.join(interp_lines)}</ul>
            Perlu dicatat: pada dataset ini, <b>tingkat kemiskinan (gabungan_pendudukmiskin)
            berkorelasi lemah</b> (|r| &lt; 0.2 di semua target) terhadap ARPS — jauh lebih lemah
            dibanding Rata-rata/Harapan Lama Sekolah (RLS/HLS) dan tenaga kerja formal. Narasi
            kebijakan sebaiknya tidak menempatkan kemiskinan sebagai faktor pendorong utama tanpa
            kualifikasi ini.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br><h3 class='section-title'>Permutation Importance (Model 16–18 Tahun / SMA)</h3>", unsafe_allow_html=True)
            imp_df = compute_permutation_importance(trained['ARPS_16to18'], df_model, 'ARPS_16to18')
            if imp_df is not None:
                fig_imp = px.bar(
                    imp_df, x='importance', y='fitur', orientation='h',
                    color='importance', color_continuous_scale='Teal', template='plotly_dark',
                )
                fig_imp.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False, yaxis_title='', xaxis_title='Penurunan R² saat fitur diacak')
                st.plotly_chart(fig_imp, use_container_width=True)
                st.caption("Dihitung dengan `sklearn.inspection.permutation_importance` pada data test 2025 (n=38) — bukan angka simulasi.")
            else:
                st.info("Data test tidak cukup untuk menghitung permutation importance.")

    # ════════════════════════════════════════════════════
    # D. EVALUASI MODEL
    # ════════════════════════════════════════════════════
    elif menu == "Evaluasi Model":
        st.markdown("<h2 class='section-title'>Evaluasi Model — Angka Asli dari Notebook</h2>", unsafe_allow_html=True)
        st.caption("Train: 2021–2024 (152 baris) · Test: 2025 (38 baris/provinsi) · split berbasis waktu, bukan random 80/20.")

        sel_target_eval = st.selectbox("Pilih Target:", TARGET_LIST, format_func=lambda t: f"{t} ({TARGET_LABELS[t]})")

        st.markdown("#### Hasil Regresi (seluruh model)")
        reg_view = REGRESI_RESULTS[REGRESI_RESULTS['target'] == sel_target_eval].sort_values('r2_test', ascending=False)
        fig_reg = px.bar(
            reg_view, x='model', y='r2_test', template='plotly_dark',
            color='r2_test', color_continuous_scale='RdYlGn', range_color=[-1, 1],
        )
        fig_reg.add_hline(y=0.5, line_dash='dash', line_color='#F39C12', annotation_text='ambang r2_test > 0.5 (syarat lolos seleksi notebook)')
        fig_reg.add_hline(y=0, line_dash='dot', line_color='#E74C3C')
        fig_reg.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_tickangle=-35, height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_reg, use_container_width=True)
        st.dataframe(
            reg_view[['model', 'r2_train', 'r2_test', 'rmse', 'mae']].style.format(
                {'r2_train': '{:.4f}', 'r2_test': '{:.4f}', 'rmse': '{:.4f}', 'mae': '{:.4f}'}
            ),
            use_container_width=True, hide_index=True,
        )

        st.markdown("#### Hasil Klasifikasi via Median Threshold (seluruh model)")
        klas_view = KLASIFIKASI_RESULTS[KLASIFIKASI_RESULTS['target'] == sel_target_eval].sort_values('f1', ascending=False)
        fig_klas = px.bar(
            klas_view, x='model', y=['accuracy', 'f1'], barmode='group', template='plotly_dark',
            color_discrete_map={'accuracy': '#3498DB', 'f1': '#E74C3C'},
        )
        fig_klas.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_tickangle=-35, yaxis_range=[0, 1], height=380)
        st.plotly_chart(fig_klas, use_container_width=True)
        st.dataframe(
            klas_view.style.format({'accuracy': '{:.4f}', 'f1': '{:.4f}'}),
            use_container_width=True, hide_index=True,
        )

        st.markdown("<div class='section-title'>Model Terbaik per Target (aturan notebook: r2_test > 0.5, lalu F1 tertinggi)</div>", unsafe_allow_html=True)
        best_tbl = pd.DataFrame([
            {'Target': t, 'Kelompok Usia': TARGET_LABELS[t], 'Model Terbaik': v['model'],
             'R² Test': v['r2_test'], 'Accuracy': v['accuracy'], 'F1': v['f1'],
             'Ensemble Learning?': 'Ya' if v['is_ensemble'] else 'Tidak — KNN (instance-based)'}
            for t, v in BEST_MODEL_PER_TARGET.items()
        ])
        st.dataframe(
            best_tbl.style.format({'R² Test': '{:.4f}', 'Accuracy': '{:.4f}', 'F1': '{:.4f}'}),
            use_container_width=True, hide_index=True,
        )
        st.markdown("""
        <div class="insight-box">
        3 dari 4 target ARPS memang paling baik diprediksi oleh model ensemble (XGBoost /
        Extra Trees). Namun untuk target <b>19–23 tahun (perguruan tinggi)</b>, model dengan
        F1 tertinggi yang lolos syarat r2_test &gt; 0.5 adalah <b>KNN Regressor</b> — model
        berbasis kemiripan tetangga terdekat, <b>bukan</b> ensemble learning. Klaim judul
        proyek yang menyebut sistem "berbasis Ensemble Learning" perlu dikualifikasi agar
        tidak generalisasi berlebihan ke seluruh sistem.
        </div>
        """, unsafe_allow_html=True)

        with st.expander("⚠️ Catatan Metodologis — keterbatasan yang BELUM diperbaiki di notebook"):
            st.markdown("""
- **Ukuran test set kecil.** Test set 2025 hanya berisi 38 baris (1 tahun × 38 provinsi).
  Semua metrik R²/accuracy/F1 di atas dihitung dari sampel sekecil ini, sehingga variansinya
  tinggi — satu-dua provinsi "sulit" bisa mengubah R² test secara drastis. Validasi tambahan
  seperti walk-forward validation (menggeser `train_end_year` dan mengulang evaluasi) akan
  membuat klaim performa jauh lebih meyakinkan bagi juri.
- **IQR-capping dihitung dari seluruh data (train+test digabung),** bukan hanya dari data
  train lalu diterapkan ke test. ini kebocoran informasi berskala kecil dari test ke train:
  batas capping "tahu" persebaran nilai tahun 2025 saat memangkas data 2021–2024. Notebook
  sudah benar menerapkan prinsip *fit scaler hanya pada train*, tapi prinsip yang sama belum
  konsisten diterapkan pada tahap outlier capping.
- **Hiperparameter regularisasi Ridge/Lasso/ElasticNet berbeda** antara pipeline regresi
  (cell 14, `alpha` 0.05–3.0) dan pipeline klasifikasi (cell 19, `alpha` 0.01–1.0) di
  notebook. Akibatnya, R² test pada tabel regresi dan Accuracy/F1 pada tabel klasifikasi
  untuk keempat model linear tersebut **bukan berasal dari objek model yang persis sama** —
  meski untuk model tree-based/ensemble/KNN (yang menentukan model terbaik di atas)
  hiperparameternya identik di kedua pipeline sehingga tidak terdampak.
- **Ambang klasifikasi memakai median historis (train), bukan target kebijakan.** "Berisiko
  tinggi" di notebook berarti "ARPS di atas median 2021–2024", bukan ambang absolut yang
  dikaitkan ke target kebijakan pendidikan nasional — cocok untuk *ranking* relatif antar
  provinsi, tapi perlu disebutkan eksplisit agar tidak dibaca sebagai ambang baku Kemendikbud/BPS.
- **Korelasi kemiskinan terhadap ARPS lemah dan berlawanan arah dari intuisi umum** (lihat
  halaman EDA & Korelasi) — narasi kebijakan berbasis kemiskinan sebagai penyebab utama
  putus sekolah perlu diuji lebih lanjut, bukan diasumsikan.
            """)

    # ════════════════════════════════════════════════════
    # E. EDA & KORELASI
    # ════════════════════════════════════════════════════
    elif menu == "EDA & Korelasi":
        st.markdown("<h2 class='section-title'>Korelasi Fitur terhadap Target ARPS</h2>", unsafe_allow_html=True)
        st.caption("Angka asli dari notebook (Pearson correlation, setelah IQR-capping) — selalu tersedia tanpa perlu dataset mentah.")

        sel_target_corr = st.selectbox("Target:", TARGET_LIST, format_func=lambda t: f"{t} ({TARGET_LABELS[t]})", key="corr_target")
        corr_view = CORR_TARGET[['fitur', sel_target_corr]].rename(columns={sel_target_corr: 'korelasi'}).sort_values('korelasi')
        fig_corr = px.bar(
            corr_view, x='korelasi', y='fitur', orientation='h', template='plotly_dark',
            color='korelasi', color_continuous_scale='RdBu', range_color=[-1, 1],
        )
        fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=460, coloraxis_showscale=False, yaxis_title='')
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("""
        <div class="caveat-box">
        Perhatikan tanda korelasi: RLS/HLS/tenaga kerja formal berkorelasi <b>negatif</b> kuat
        (capaian pendidikan/pasar kerja formal makin tinggi → ARPS makin rendah, sesuai
        intuisi). Sebaliknya, TPT dan NEET juga berkorelasi negatif terhadap ARPS di dataset
        ini — kebalikan dari intuisi "pengangguran tinggi → putus sekolah tinggi". Ini
        kemungkinan mencerminkan struktur pasar kerja informal di wilayah seperti Papua (TPT
        rendah bukan karena makmur, tapi karena sangat sedikit yang berstatus aktif mencari
        kerja formal). Gunakan angka ini untuk melatih model, bukan untuk klaim kausal langsung.
        </div>
        """, unsafe_allow_html=True)

        if df_model is not None:
            st.markdown("<div class='section-title'>Distribusi Variabel (data panel 2021–2025)</div>", unsafe_allow_html=True)
            sel_var = st.selectbox("Pilih variabel:", FITUR + TARGET_LIST, key="dist_var")
            fig_hist = px.histogram(
                df_model, x=sel_var, nbins=30, template='plotly_dark', marginal='box',
                color_discrete_sequence=['#E67E22'],
            )
            fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("<div class='section-title'>Scatter: Fitur vs ARPS</div>", unsafe_allow_html=True)
            cx, cy = st.columns(2)
            x_var = cx.selectbox("Sumbu X:", FITUR, index=FITUR.index('gabungan_RLS'))
            y_var = cy.selectbox("Sumbu Y:", TARGET_LIST, index=2)
            fig_sc = px.scatter(
                df_model, x=x_var, y=y_var, color='Tahun', hover_data=['Provinsi', 'Tahun'],
                template='plotly_dark', trendline='ols', trendline_color_override='#FFD700',
            )
            fig_sc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info(
                "📊 Histogram distribusi & scatter per baris tahun butuh dataset panel mentah — "
                "unggah `Dataset_Gabungan_Fix.xlsx` lewat sidebar untuk mengaktifkan."
            )

    # ════════════════════════════════════════════════════
    # F. TENTANG PROYEK
    # ════════════════════════════════════════════════════
    elif menu == "Tentang Proyek":
        st.markdown("<h2 class='section-title'>Tentang DropAlert</h2>", unsafe_allow_html=True)
        st.markdown("""
**DropAlert** adalah sistem eksplorasi & prediksi risiko putus sekolah di Indonesia,
menggabungkan (1) perbandingan berbagai algoritma regresi untuk memprediksi Angka Risiko
Putus Sekolah (ARPS) per kelompok usia, (2) klasifikasi biner berbasis ambang median historis,
dan (3) K-Means clustering tingkat provinsi untuk pemetaan prioritas intervensi.

### Latar Belakang
Partisipasi sekolah di sejumlah kelompok usia dan wilayah di Indonesia masih menunjukkan
kesenjangan. Dashboard ini dibangun untuk membantu memvisualisasikan pola tersebut
berdasarkan indikator BPS 2021–2025, dan mendukung diskusi prioritas intervensi pendidikan
per provinsi — bukan sebagai alat penentu kebijakan tunggal.

### Pendekatan Metodologi
| Aspek | Detail |
|---|---|
| Sumber Data | Badan Pusat Statistik (BPS), panel provinsi × tahun, 2021–2025 |
| Fitur (X) | 14 indikator: kemiskinan, TPT, NEET, tenaga kerja formal, HLS, RLS, rasio murid/guru & murid/sekolah (SD/SMP/SMA/SMK) |
| Target (Y) | ARPS = 100 − APS, untuk kelompok usia 7–12, 13–15, 16–18, 19–23 |
| Split Data | Time-series — Train 2021–2024 (152 baris), Test 2025 (38 baris) |
| Model dibandingkan | Linear/Ridge/Lasso/ElasticNet, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, KNN Regressor |
| Klasifikasi | Ambang median ARPS dari data TRAIN, dievaluasi dengan Accuracy & F1 |
| Clustering | K-Means (k=3, dipilih via silhouette score) atas 14 fitur yang sama, dirata-rata per provinsi |

### Model Terbaik per Target (lihat halaman Evaluasi Model untuk rincian & catatan metodologis)
        """)
        best_tbl2 = pd.DataFrame([
            {'Target': t, 'Model': v['model'], 'R² Test': f"{v['r2_test']:.3f}",
             'Accuracy': f"{v['accuracy']:.3f}", 'F1': f"{v['f1']:.3f}"}
            for t, v in BEST_MODEL_PER_TARGET.items()
        ])
        st.dataframe(best_tbl2, use_container_width=True, hide_index=True)

        st.markdown("""
### Keterbatasan yang Diketahui
- Test set kecil (n=38, satu tahun) → interval kepercayaan metrik lebar.
- IQR-capping dihitung dari data gabungan train+test (lihat expander di halaman Evaluasi Model).
- Klasifikasi memakai ambang median historis, bukan ambang kebijakan absolut.
- Korelasi kemiskinan terhadap ARPS lemah pada dataset ini — jangan disimpulkan sebagai
  penyebab tunggal putus sekolah.
        """)

        st.markdown("---")
        st.markdown("**Cara Menjalankan Aplikasi**")
        st.code("streamlit run app.py", language="bash")
        st.markdown(
            f"File `{CLUSTER_CSV_PATH}` wajib berada di folder yang sama dengan `app.py`. "
            "Untuk mengaktifkan prediksi interaktif & EDA per-tahun, sediakan "
            "`Dataset_Gabungan_Fix.xlsx` di folder yang sama atau unggah lewat sidebar."
        )

    # ════════════════════════════════════════════════════
    # FOOTER
    # ════════════════════════════════════════════════════
    st.markdown("""
    <div style="margin-top: 5rem; padding: 1.5rem 2rem; background: linear-gradient(90deg, #3D1010 0%, #800000 100%); border-radius: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; border-left: 5px solid #F39C12; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
         <div>
            <p style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin: 0;">DropAlert</p>
            <p style="font-size: 0.85rem; color: #FFD5CC; margin: 0;">Deteksi Risiko Putus Sekolah & Pemetaan Klaster Provinsi · Indonesia</p>
        </div>
        <div style="font-size: 0.85rem; color: #FFD5CC; text-align: right; line-height: 1.6;">
            Sumber Data: BPS (2021–2025)<br>
            IT FEST 6.0 — Lomba Karya Tulis Ilmiah
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# ENTRYPOINT
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    build_dashboard()
