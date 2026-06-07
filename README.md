# SmartRetain — Employee Attrition Intelligence

> Prediksi risiko attrition karyawan berbasis Machine Learning sebagai solusi HR berbasis data.
> 
> **Coding Camp 2026 powered by DBS Foundation** | CC26-PRU470 | Future-Ready Work & Economy

---

## Deskripsi Proyek

SmartRetain adalah aplikasi dashboard interaktif yang membantu divisi HR mengidentifikasi karyawan yang berpotensi resign **sebelum** keputusan itu terjadi. Dibangun menggunakan Python dan Streamlit, aplikasi ini menggabungkan analisis data eksploratif dan model Machine Learning (Random Forest & KNN) untuk menghasilkan **Attrition Risk Scoring** per karyawan.

---

## Fitur Aplikasi

| Halaman | Fitur |
|---|---|
| **Overview** | Ringkasan metrik utama, distribusi attrition, business questions, alur analisis |
| **Exploratory Analysis** | Visualisasi distribusi usia, gaji, rating, promosi, heatmap korelasi, risk scoring |
| **Model & Evaluasi** | Classification report, feature importance, confusion matrix RF vs KNN |
| **Prediksi Risiko** | Input data karyawan → skor risiko 0–100 (High/Medium/Low Risk) |
| **A/B Testing** | 5-fold cross-validation + paired t-test RF vs KNN |

---

## Struktur File
smartretain/
-app.py (File utama aplikasi Streamlit)
-model_rf.pkl (Model Random Forest yang sudah dilatih)
-feature_columns.json (Daftar fitur yang digunakan model)
-train_data.csv (Dataset (unduh dari Kaggle))
-requirements.txt (Library yang dibutuhkan)
-predicting_Employee_Attrition.ipynb
-README.md (Dokumentasi)


---

## Cara Menjalankan Secara Lokal

### 1. Clone repository ini
```bash
github.com/sayesaye45/Smart-Retain
cd smartretain
```

### 2. Install library yang dibutuhkan
```bash
pip install -r requirements.txt
```

### 3. Unduh dataset
Unduh dataset dari Kaggle:
https://www.kaggle.com/datasets/pavan9065/predictingemployeeattrition

Simpan file `train_data.csv` di folder yang sama dengan `app.py`

### 4. Pastikan semua file ada

### 5. Jalankan aplikasi
```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser: `http://localhost:8501`

---

## Requirements
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy

---

## Dataset

- **Sumber:** Kaggle — Predicting Employee Attrition (pavan9065)
- **Jumlah data:** 19.104 baris → 16.695 setelah cleaning
- **Fitur:** 13 kolom (Age, Gender, Education Level, Salary, Designation, dll)
- **Target:** Attrition (0 = Aktif, 1 = Keluar)

---

## Model & Hasil

| Model | Akurasi | Keterangan |
|---|---|---|
| Random Forest | 84.1% | Model final (`class_weight='balanced'`) |
| KNN (k=5) | Pembanding | Digunakan untuk A/B Testing |

**Faktor paling berpengaruh (Feature Importance RF):**
1. Total Business Value
2. Salary
3. Quarterly Rating

## Link Drive Model AI
https://drive.google.com/drive/folders/1e22AgFCKp_NyC_4D5hDFBTQJCS-suyox?usp=sharing

---

##  Risk Scoring System

| Kategori | Skor | Rekomendasi HR |
|---|---|---|
| 🔴 High Risk | ≥ 70 | Intervensi segera |
| 🟡 Medium Risk | 40–69 | Pantau & program engagement |
| 🟢 Low Risk | < 40 | Pertahankan kondisi kerja |

---

##  Dikerjakan oleh

**Kurnia Irianti** — Data Scientist  
CDCC228D6X2404 | Coding Camp 2026 powered by DBS Foundation
