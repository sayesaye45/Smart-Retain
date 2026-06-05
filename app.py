import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import collections
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="SmartRetain — Employee Attrition Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.stApp { background: #07070f; color: #e8e8f0; }

[data-testid="stSidebar"] {
    background: #0d0d1a !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #c8c8d8 !important; }

[data-testid="metric-container"] {
    background: #111120;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #a594f9 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #6b6b88 !important;
}
[data-testid="stTabs"] button {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #6b6b88 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #a594f9 !important;
    border-bottom: 2px solid #a594f9 !important;
}
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input {
    background: #111120 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8e8f0 !important;
}
.stButton > button {
    background: #a594f9 !important;
    color: #07070f !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.5px !important;
    padding: 0.6rem 1.5rem !important;
}
.stButton > button:hover {
    background: #c3b8ff !important;
    box-shadow: 0 6px 20px rgba(165,148,249,0.3) !important;
}
hr { border-color: rgba(255,255,255,0.06) !important; }
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #f0f0fa;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}
.page-subtitle {
    font-size: 0.9rem;
    color: #6b6b88;
    margin-bottom: 1.5rem;
    font-weight: 300;
}
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #a594f9;
    margin-bottom: 0.8rem;
}
.risk-card-high {
    background: linear-gradient(135deg, rgba(255,92,92,0.12), rgba(255,92,92,0.04));
    border: 1px solid rgba(255,92,92,0.25);
    border-radius: 16px; padding: 1.5rem; text-align: center;
}
.risk-card-mid {
    background: linear-gradient(135deg, rgba(245,166,35,0.12), rgba(245,166,35,0.04));
    border: 1px solid rgba(245,166,35,0.25);
    border-radius: 16px; padding: 1.5rem; text-align: center;
}
.risk-card-low {
    background: linear-gradient(135deg, rgba(62,207,142,0.12), rgba(62,207,142,0.04));
    border: 1px solid rgba(62,207,142,0.25);
    border-radius: 16px; padding: 1.5rem; text-align: center;
}
.insight-box {
    background: rgba(165,148,249,0.06);
    border-left: 3px solid #a594f9;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem; margin: 0.8rem 0;
    font-size: 0.88rem; color: #c8c8d8; line-height: 1.6;
}
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #9090a8 !important;
    border: none !important;
    border-radius: 8px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 400 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0 !important;
    padding: 0.5rem 0.85rem !important;
    box-shadow: none !important;
    margin: 1px 0 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.05) !important;
    color: #e8e8f0 !important;
    box-shadow: none !important;
    transform: none !important;
}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    'figure.facecolor': '#111120', 'axes.facecolor': '#111120',
    'axes.edgecolor': '#2a2a3f', 'axes.labelcolor': '#9090a8',
    'xtick.color': '#6b6b88', 'ytick.color': '#6b6b88',
    'text.color': '#c8c8d8', 'grid.color': '#1e1e2e', 'grid.alpha': 0.6,
})
ACCENT = '#a594f9'
HIGH_C = '#ff5c5c'
MID_C  = '#f5a623'
LOW_C  = '#3ecf8e'
FEATURE_COLS = ['Age','Gender','Education_Level','Salary',
                'Joining Designation','Designation',
                'Total Business Value','Quarterly Rating',
                'age_category','salary_category','promotion_level']

@st.cache_data
def load_and_process():
    data = pd.read_csv("train_data.csv")
    data['Attrition'] = data['LastWorkingDate'].apply(lambda x: 1 if pd.notnull(x) else 0)
    cols = ['Age','Gender','Education_Level','Salary','Joining Designation','Designation',
            'Total Business Value','Quarterly Rating','Attrition']
    d = data[cols].copy().drop_duplicates()
    for col in ['Age','Salary','Total Business Value','Quarterly Rating']:
        d[col] = d[col].fillna(d[col].median())
    for col in ['Gender','Education_Level','Joining Designation','Designation']:
        d[col] = d[col].fillna(d[col].mode()[0])
    d['Gender'] = d['Gender'].map({'Male':1,'Female':0})
    d['Education_Level'] = d['Education_Level'].map({'College':1,'Bachelor':2,'Master':3})
    d['age_category']    = pd.cut(d['Age'], bins=[0,25,35,45,100], labels=[0,1,2,3]).astype(int)
    d['salary_category'] = pd.cut(d['Salary'], bins=[0,30000,60000,90000,float('inf')], labels=[0,1,2,3]).astype(int)
    d['promotion_level'] = d['Designation'] - d['Joining Designation']
    return d

@st.cache_resource
def load_rf_model():
    with open("model_rf.pkl","rb") as f:
        return pickle.load(f)

@st.cache_resource
def train_knn_and_split(_data):
    X = _data[FEATURE_COLS]
    y = _data['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    return knn, X_train, X_test, y_train, y_test, X, y

try:
    data_new  = load_and_process()
    model_rf  = load_rf_model()
    model_knn, X_train, X_test, y_train, y_test, X, y = train_knn_and_split(data_new)
except Exception as e:
    st.error(f"Gagal memuat file: {e}")
    st.info("Pastikan `train_data.csv` dan `model_rf.pkl` ada di folder yang sama dengan `app.py`")
    st.stop()

def kategorikan_risiko(score):
    if score >= 70:   return 'High Risk'
    elif score >= 40: return 'Medium Risk'
    else:             return 'Low Risk'

# ── SIDEBAR ──────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1.5rem'>
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:0.3rem'>
            <div style='width:32px;height:32px;background:linear-gradient(135deg,#a594f9,#7c6fef);
            border-radius:9px;display:flex;align-items:center;justify-content:center;
            font-size:16px;flex-shrink:0'>◈</div>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;
            color:#f0f0fa;letter-spacing:-0.3px'>SmartRetain</div>
        </div>
        <div style='font-size:0.72rem;color:#4a4a6a;padding-left:42px'>Employee Attrition Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if "active_page" not in st.session_state:
        st.session_state.active_page = "Overview"

    for icon, label in [("▣","Overview"),("◎","Exploratory Analysis"),
                        ("◈","Model & Evaluasi"),("◉","Prediksi Risiko"),("⊞","A/B Testing")]:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.active_page = label
            st.rerun()

    page = st.session_state.active_page
    st.markdown("---")
    total  = len(data_new)
    keluar = int(data_new['Attrition'].sum())
    st.markdown("<div style='font-size:0.72rem;color:#4a4a6a;text-transform:uppercase;letter-spacing:1px'>Dataset</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.85rem;color:#c8c8d8;margin-top:0.4rem'>{total:,} records · {keluar:,} attrition</div>", unsafe_allow_html=True)

# ── PAGE: OVERVIEW ────────────────────────────
if page == "Overview":
    st.markdown('<div class="page-title">Dashboard Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Ringkasan proyek SmartRetain - Prediksi Employee Attrition berbasis Machine Learning</div>', unsafe_allow_html=True)

    total  = len(data_new)
    keluar = int(data_new['Attrition'].sum())
    aktif  = total - keluar
    rate   = keluar / total * 100
    acc_rf = accuracy_score(y_test, model_rf.predict(X_test))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Karyawan",   f"{total:,}")
    c2.metric("Karyawan Keluar",  f"{keluar:,}")
    c3.metric("Attrition Rate",   f"{rate:.1f}%")
    c4.metric("Akurasi Model RF", f"{acc_rf*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-label">Distribusi Attrition</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(['Aktif','Keluar'], [aktif, keluar],
                      color=[ACCENT, HIGH_C], width=0.45, edgecolor='none', zorder=3)
        for bar, val in zip(bars, [aktif, keluar]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
                    f'{val:,}', ha='center', fontsize=11, fontweight='bold', color='#e8e8f0')
        ax.set_ylabel('Jumlah Karyawan', fontsize=9)
        ax.set_title('Karyawan Aktif vs Keluar', fontsize=11, color='#e8e8f0', pad=12, fontweight='bold')
        ax.grid(axis='y', zorder=0)
        ax.spines[['top','right','left','bottom']].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-label">Business Questions</div>', unsafe_allow_html=True)
        for label, q in [
            ("BQ 1","Faktor apa yang paling mempengaruhi keputusan karyawan untuk keluar?"),
            ("BQ 2","Apakah karyawan bergaji rendah & rating rendah lebih cenderung keluar?"),
            ("BQ 3","Seberapa akurat model ML dalam memprediksi attrition?"),
        ]:
            st.markdown(f"""
            <div style='background:#111120;border:1px solid rgba(165,148,249,0.15);
            border-radius:12px;padding:0.9rem 1rem;margin-bottom:0.6rem;display:flex;gap:0.8rem;align-items:flex-start'>
                <div style='background:#a594f9;color:#07070f;font-family:Syne,sans-serif;
                font-weight:800;font-size:0.65rem;padding:0.2rem 0.5rem;border-radius:6px;
                white-space:nowrap;margin-top:0.1rem'>{label}</div>
                <div style='font-size:0.82rem;color:#c8c8d8;line-height:1.5'>{q}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Alur Analisis</div>', unsafe_allow_html=True)
    steps = ["Pengumpulan Data","Penelaahan Data","Validasi Data","Penentuan Variabel",
             "Cleaning Data","Feature Engineering","Visualisasi & EDA","Modelling","Evaluasi & Risk Scoring"]
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            st.markdown(f"""
            <div style='background:#111120;border:1px solid rgba(165,148,249,0.15);
            border-radius:10px;padding:0.7rem 0.5rem;text-align:center'>
                <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:800;color:#a594f9'>{i+1}</div>
                <div style='font-size:0.65rem;color:#6b6b88;margin-top:0.3rem;line-height:1.4'>{step}</div>
            </div>""", unsafe_allow_html=True)

# ── PAGE: EDA ─────────────────────────────────
elif page == "Exploratory Analysis":
    st.markdown('<div class="page-title">Exploratory Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Menggali insight dari data untuk menjawab business questions</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Distribusi","Gaji & Rating","Korelasi","Promosi & Level Gaji","Risk Scoring"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-label">Distribusi Usia Karyawan</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(data_new['Age'], bins=20, color=ACCENT, alpha=0.85, edgecolor='none')
            ax.axvline(data_new['Age'].median(), color=HIGH_C, linestyle='--',
                       linewidth=1.5, label=f"Median: {data_new['Age'].median():.0f} thn")
            ax.set_xlabel('Usia'); ax.set_ylabel('Frekuensi')
            ax.set_title('Distribusi Usia Karyawan', color='#e8e8f0', fontweight='bold')
            ax.legend(fontsize=8); ax.grid(axis='y')
            ax.spines[['top','right']].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('<div class="insight-box">Mayoritas karyawan berusia 25–35 tahun, kelompok paling mobile dalam karir dan paling berisiko untuk keluar.</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-label">Jumlah Karyawan Aktif vs Keluar</div>', unsafe_allow_html=True)
            attrition_counts = data_new['Attrition'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(5, 3.5))
            bars = ax.bar(['Aktif (0)','Keluar (1)'], attrition_counts.values,
                          color=[ACCENT, HIGH_C], width=0.45, edgecolor='none')
            for bar, val in zip(bars, attrition_counts.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                        str(val), ha='center', fontsize=10, fontweight='bold', color='#e8e8f0')
            ax.set_xlabel('Status Karyawan'); ax.set_ylabel('Jumlah')
            ax.set_title('Jumlah Karyawan Aktif vs Keluar', color='#e8e8f0', fontweight='bold')
            ax.grid(axis='y'); ax.spines[['top','right']].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('<div class="insight-box">Terdapat class imbalance yang signifikan antara 90.3% aktif vs 9.7% keluar. Kondisi ini wajar di data HR nyata dan sudah ditangani menggunakan <b>class_weight=balanced</b> saat training model.</div>', unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-label">Distribusi Gaji per Status Attrition</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            bp = ax.boxplot(
                [data_new[data_new['Attrition']==0]['Salary'],
                 data_new[data_new['Attrition']==1]['Salary']],
                patch_artist=True,
                medianprops={'color':'#ffffff','linewidth':2.5},
                whiskerprops={'color':'#4a4a6a','linewidth':1.2},
                capprops={'color':'#4a4a6a','linewidth':1.2},
                flierprops={'marker':'o','markerfacecolor':'#4a4a6a',
                            'markeredgecolor':'none','markersize':2,'alpha':0.3},
                boxprops={'linewidth':1.2}
            )
            bp['boxes'][0].set_facecolor(ACCENT+'55')
            bp['boxes'][1].set_facecolor(HIGH_C+'55')
            ax.set_xticks([1,2]); ax.set_xticklabels(['Aktif (0)','Keluar (1)'])
            ax.set_ylabel('Gaji')
            ax.set_title('Distribusi Gaji Berdasarkan Status Attrition', color='#e8e8f0', fontweight='bold')
            ax.grid(axis='y'); ax.spines[['top','right']].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('<div class="insight-box"><b>Menjawab BQ 2 (gaji):</b> Median gaji karyawan yang keluar lebih rendah. Kompensasi adalah faktor signifikan dalam keputusan resign.</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-label">Attrition Rate per Quarterly Rating</div>', unsafe_allow_html=True)
            rate_rating = data_new.groupby('Quarterly Rating')['Attrition'].mean() * 100
            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors_r = [HIGH_C if v >= rate_rating.mean() else ACCENT for v in rate_rating.values]
            bars = ax.bar(rate_rating.index.astype(str), rate_rating.values,
                          color=colors_r, width=0.5, edgecolor='none')
            for bar, val in zip(bars, rate_rating.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                        f'{val:.1f}%', ha='center', fontsize=9, color='#e8e8f0')
            ax.set_xlabel('Quarterly Rating (1=Terendah, 4=Tertinggi)')
            ax.set_ylabel('Attrition Rate (%)')
            ax.set_title('Attrition Rate per Quarterly Rating', color='#e8e8f0', fontweight='bold')
            ax.grid(axis='y'); ax.spines[['top','right']].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('<div class="insight-box"><b>Menjawab BQ 2 (rating):</b> Rating rendah (1–2) berkorelasi dengan attrition rate lebih tinggi. Karyawan yang performanya kurang baik lebih mudah resign.</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-label">Heatmap Korelasi Antar Variabel</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = data_new.select_dtypes(include='number').corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    linewidths=0.5, linecolor='#07070f', ax=ax,
                    annot_kws={'size':8}, cbar_kws={'shrink':0.8})
        ax.set_title('Heatmap Korelasi Antar Variabel', color='#e8e8f0', fontweight='bold', pad=12, fontsize=12)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box">Salary dan Quarterly Rating menunjukkan korelasi negatif dengan Attrition semakin tinggi keduanya, semakin kecil risiko karyawan keluar. Tidak ada multikolinearitas tinggi antar fitur, artinya setiap fitur memberikan informasi yang unik.</div>', unsafe_allow_html=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-label">Attrition Rate per Level Promosi</div>', unsafe_allow_html=True)
            rate_promo = data_new.groupby('promotion_level')['Attrition'].mean() * 100
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(rate_promo.index, rate_promo.values, color=ACCENT,
                    linewidth=2.5, marker='o', markerfacecolor=HIGH_C,
                    markeredgecolor='none', markersize=7)
            ax.fill_between(rate_promo.index, rate_promo.values, alpha=0.15, color=ACCENT)
            ax.set_xlabel('Promotion Level (0 = tidak pernah naik jabatan)')
            ax.set_ylabel('Attrition Rate (%)')
            ax.set_title('Attrition Rate per Level Promosi', color='#e8e8f0', fontweight='bold')
            ax.grid(axis='y'); ax.spines[['top','right']].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('<div class="insight-box">Karyawan yang tidak pernah naik jabatan (promotion_level = 0) memiliki attrition rate tertinggi. Karir stagnan mendorong karyawan mencari peluang lain.</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-label">Attrition Rate per Level Gaji</div>', unsafe_allow_html=True)
            sal_labels = {0:'Low',1:'Medium',2:'High',3:'Very High'}
            rate_sal = data_new.groupby('salary_category')['Attrition'].mean() * 100
            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors_s = [HIGH_C if v >= rate_sal.mean() else LOW_C for v in rate_sal.values]
            bars = ax.bar([sal_labels[i] for i in rate_sal.index], rate_sal.values,
                          color=colors_s, width=0.5, edgecolor='none')
            for bar, val in zip(bars, rate_sal.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                        f'{val:.1f}%', ha='center', fontsize=9, color='#e8e8f0')
            ax.set_ylabel('Attrition Rate (%)')
            ax.set_title('Attrition Rate per Level Gaji', color='#e8e8f0', fontweight='bold')
            ax.grid(axis='y'); ax.spines[['top','right']].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('<div class="insight-box">Karyawan di level gaji rendah (Low) memiliki attrition rate paling tinggi, semakin tinggi level gaji semakin kecil risiko karyawan keluar.</div>', unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="section-label">Distribusi Risk Category Karyawan</div>', unsafe_allow_html=True)
        risk_probs = model_rf.predict_proba(X_test)[:, 1]
        risk_scores_all = (risk_probs * 100).round(1)
        risk_cats = [kategorikan_risiko(s) for s in risk_scores_all]
        risk_counts = collections.Counter(risk_cats)
        ordered_keys = ['High Risk','Medium Risk','Low Risk']
        ordered_vals = [risk_counts.get(k, 0) for k in ordered_keys]

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            bars = ax.bar(ordered_keys, ordered_vals,
                          color=[HIGH_C, MID_C, LOW_C], width=0.5, edgecolor='none')
            for bar, val in zip(bars, ordered_vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                        str(val), ha='center', fontsize=10, fontweight='bold', color='#e8e8f0')
            ax.set_ylabel('Jumlah Karyawan')
            ax.set_title('Distribusi Risk Category Karyawan', color='#e8e8f0', fontweight='bold')
            ax.grid(axis='y'); ax.spines[['top','right']].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            total_test = len(risk_scores_all)
            for cat, val, color in zip(ordered_keys, ordered_vals, [HIGH_C, MID_C, LOW_C]):
                pct = val / total_test * 100
                st.markdown(f"""
                <div style='background:#111120;border:1px solid rgba(255,255,255,0.07);
                border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.5rem;
                display:flex;justify-content:space-between;align-items:center'>
                    <div style='font-family:Syne,sans-serif;font-weight:700;
                    font-size:0.85rem;color:{color}'>{cat}</div>
                    <div>
                        <span style='font-family:Syne,sans-serif;font-weight:800;
                        font-size:1.1rem;color:{color}'>{val}</span>
                        <span style='font-size:0.75rem;color:#6b6b88;margin-left:4px'>({pct:.1f}%)</span>
                    </div>
                </div>""", unsafe_allow_html=True)
        st.markdown('<div class="insight-box">Sistem Risk Scoring mengkategorikan setiap karyawan ke dalam tiga level risiko berdasarkan probabilitas prediksi model. Karyawan High Risk adalah prioritas utama yang perlu mendapat perhatian segera dari tim HR.</div>', unsafe_allow_html=True)

# ── PAGE: MODEL & EVALUASI ────────────────────
elif page == "Model & Evaluasi":
    st.markdown('<div class="page-title">Model & Evaluasi</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Perbandingan performa Random Forest vs KNN</div>', unsafe_allow_html=True)

    y_pred_rf  = model_rf.predict(X_test)
    y_pred_knn = model_knn.predict(X_test)
    acc_rf  = accuracy_score(y_test, y_pred_rf)
    acc_knn = accuracy_score(y_test, y_pred_knn)

    c1, c2 = st.columns(2)
    c1.metric("Akurasi Random Forest", f"{acc_rf*100:.2f}%")
    c2.metric("Akurasi KNN",           f"{acc_knn*100:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Classification Report","Feature Importance","Confusion Matrix"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-label">Random Forest</div>', unsafe_allow_html=True)
            rf_rep = pd.DataFrame(classification_report(y_test, y_pred_rf, output_dict=True)).T.round(3)
            st.dataframe(rf_rep.style.background_gradient(cmap='Purples', axis=None), use_container_width=True)
        with col2:
            st.markdown('<div class="section-label">KNN</div>', unsafe_allow_html=True)
            knn_rep = pd.DataFrame(classification_report(y_test, y_pred_knn, output_dict=True)).T.round(3)
            st.dataframe(knn_rep.style.background_gradient(cmap='Purples', axis=None), use_container_width=True)
        st.markdown('<div class="insight-box"><b>Menjawab BQ 3:</b> Perhatikan nilai <b>recall kelas 1 (keluar)</b>, ini metrik terpenting. Random Forest dengan <code>class_weight=balanced</code> lebih sensitif mendeteksi karyawan berisiko dibanding KNN.</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-label">Feature Importance — Menjawab BQ 1</div>', unsafe_allow_html=True)
        fi = pd.Series(model_rf.feature_importances_, index=FEATURE_COLS).sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_fi = [HIGH_C if v >= fi.max()*0.75 else ACCENT for v in fi.values]
        ax.barh(fi.index, fi.values, color=colors_fi, height=0.6, edgecolor='none')
        for i, (idx, val) in enumerate(fi.items()):
            ax.text(val+0.001, i, f'{val:.3f}', va='center', fontsize=8, color='#c8c8d8')
        ax.set_xlabel('Importance Score')
        ax.set_xlim(0, fi.max()*1.18)
        ax.set_title('Kontribusi Setiap Fitur dalam Prediksi Attrition',
                     color='#e8e8f0', fontweight='bold', pad=12)
        ax.grid(axis='x'); ax.spines[['top','right']].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="insight-box"><b>Menjawab BQ 1:</b> Total Business Value, Salary, dan Quarterly Rating adalah tiga faktor paling berpengaruh, mengkonfirmasi bahwa faktor finansial dan performa mendominasi keputusan resign.</div>', unsafe_allow_html=True)

    with tab3:
        col1, col2 = st.columns(2)
        for col, preds, title in [(col1, y_pred_rf,"Random Forest"),(col2, y_pred_knn,"KNN")]:
            with col:
                st.markdown(f'<div class="section-label">Confusion Matrix — {title}</div>', unsafe_allow_html=True)
                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots(figsize=(4, 3.5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                            linewidths=1, linecolor='#07070f',
                            xticklabels=['Aktif','Keluar'],
                            yticklabels=['Aktif','Keluar'],
                            annot_kws={'size':12,'weight':'bold'}, ax=ax)
                ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
                ax.set_title(f'Confusion Matrix — {title}', color='#e8e8f0', fontweight='bold', fontsize=10)
                fig.tight_layout(); st.pyplot(fig); plt.close()

# ── PAGE: PREDIKSI RISIKO ─────────────────────
elif page == "Prediksi Risiko":
    st.markdown('<div class="page-title">Prediksi Risiko Attrition</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Masukkan data karyawan untuk mendapatkan skor risiko berbasis model Random Forest</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown('<div class="section-label">Data Karyawan</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age           = st.number_input("Usia", 18, 60, 28)
            salary        = st.number_input("Gaji", 10000, 500000, 57000, step=1000)
            tbv           = st.number_input("Total Business Value", -100000, 500000, 15000, step=1000)
            joining_desig = st.selectbox("Jabatan Awal (1–5)", [1,2,3,4,5])
        with c2:
            gender    = st.selectbox("Jenis Kelamin", ["Male","Female"])
            edu       = st.selectbox("Pendidikan", ["College","Bachelor","Master"])
            rating    = st.selectbox("Quarterly Rating", [1,2,3,4])
            desig_now = st.selectbox("Jabatan Saat Ini (1–5)", [1,2,3,4,5])
        predict_btn = st.button("Analisis Risiko Attrition", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-label">Hasil Analisis</div>', unsafe_allow_html=True)
        if predict_btn:
            gender_enc = 1 if gender == "Male" else 0
            edu_enc    = {"College":1,"Bachelor":2,"Master":3}[edu]
            age_cat    = 0 if age < 25 else (1 if age < 35 else (2 if age < 45 else 3))
            sal_cat    = 0 if salary < 30000 else (1 if salary < 60000 else (2 if salary < 90000 else 3))
            promo      = desig_now - joining_desig
            features   = np.array([[age, gender_enc, edu_enc, salary,
                                     joining_desig, desig_now, tbv,
                                     rating, age_cat, sal_cat, promo]])
            prob  = model_rf.predict_proba(features)[0][1]
            score = round(prob * 100, 1)

            if score >= 70:
                cat, card, color, emoji = "High Risk","risk-card-high",HIGH_C,"🔴"
                desc = "Karyawan ini memiliki probabilitas tinggi untuk keluar. HR disarankan segera melakukan evaluasi kompensasi, jalur karir, dan kondisi kerja."
            elif score >= 40:
                cat, card, color, emoji = "Medium Risk","risk-card-mid",MID_C,"🟡"
                desc = "Perlu dipantau secara berkala. Pertimbangkan program engagement atau diskusi karir untuk menjaga motivasi."
            else:
                cat, card, color, emoji = "Low Risk","risk-card-low",LOW_C,"🟢"
                desc = "Karyawan ini relatif stabil. Pertahankan kondisi kerja yang ada dan pastikan apresiasi diberikan secara konsisten."

            st.markdown(f"""
            <div class="{card}" style="margin-bottom:1rem">
                <div style="font-family:Syne,sans-serif;font-size:0.7rem;font-weight:700;
                text-transform:uppercase;letter-spacing:2px;color:#6b6b88;margin-bottom:0.5rem">Risk Score</div>
                <div style="font-family:Syne,sans-serif;font-size:3.5rem;font-weight:800;
                color:{color};line-height:1">{score}<span style="font-size:1.5rem">%</span></div>
                <div style="font-family:Syne,sans-serif;font-size:0.9rem;font-weight:700;
                color:{color};margin:0.8rem 0">{emoji} {cat}</div>
                <div style="font-size:0.82rem;color:#9090a8;line-height:1.5">{desc}</div>
            </div>""", unsafe_allow_html=True)

            summary = {"Usia":age,"Gender":gender,"Pendidikan":edu,
                       "Gaji":f"{salary:,}","Rating":rating,
                       "Jabatan":f"L{joining_desig} → L{desig_now}",
                       "Promotion Level":promo,"TBV":f"{tbv:,}"}
            st.dataframe(pd.DataFrame(list(summary.items()), columns=["Atribut","Nilai"]),
                         use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div style='background:#111120;border:1px dashed rgba(165,148,249,0.2);
            border-radius:16px;padding:2.5rem;text-align:center;color:#4a4a6a'>
                <div style='font-size:2rem;margin-bottom:0.8rem'>📋</div>
                <div style='font-family:Syne,sans-serif;font-size:0.85rem'>
                Isi data karyawan di sebelah kiri<br>lalu klik Analisis Risiko Attrition</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Panduan Kategori Risiko</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="risk-card-high">
        <div style='font-family:Syne,sans-serif;font-weight:800;color:#ff5c5c'>🔴 High Risk</div>
        <div style='font-size:1.8rem;font-weight:800;color:#ff5c5c;font-family:Syne,sans-serif;margin:0.4rem 0'>≥ 70</div>
        <div style='font-size:0.78rem;color:#9090a8'>Intervensi segera. Evaluasi kompensasi dan jalur karir.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="risk-card-mid">
        <div style='font-family:Syne,sans-serif;font-weight:800;color:#f5a623'>🟡 Medium Risk</div>
        <div style='font-size:1.8rem;font-weight:800;color:#f5a623;font-family:Syne,sans-serif;margin:0.4rem 0'>40–69</div>
        <div style='font-size:0.78rem;color:#9090a8'>Pantau dan lakukan program engagement.</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="risk-card-low">
        <div style='font-family:Syne,sans-serif;font-weight:800;color:#3ecf8e'>🟢 Low Risk</div>
        <div style='font-size:1.8rem;font-weight:800;color:#3ecf8e;font-family:Syne,sans-serif;margin:0.4rem 0'>< 40</div>
        <div style='font-size:0.78rem;color:#9090a8'>Karyawan stabil. Pertahankan kondisi kerja.</div>
        </div>""", unsafe_allow_html=True)

# ── PAGE: A/B TESTING ─────────────────────────
elif page == "A/B Testing":
    st.markdown('<div class="page-title">A/B Testing</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Uji statistik untuk membuktikan apakah perbedaan performa kedua model signifikan</div>', unsafe_allow_html=True)

    st.markdown('<div class="insight-box">Membandingkan akurasi dua model secara langsung belum cukup, bisa saja perbedaannya terjadi karena kebetulan. A/B Testing menggunakan <b>5-fold cross-validation</b> dan <b>paired t-test</b> untuk membuktikan signifikansi statistiknya.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner("Menjalankan cross-validation..."):
        scores_rf  = cross_val_score(model_rf,  X, y, cv=5, scoring='accuracy')
        scores_knn = cross_val_score(model_knn, X, y, cv=5, scoring='accuracy')
        t_stat, p_value = stats.ttest_rel(scores_rf, scores_knn)

    c1, c2, c3 = st.columns(3)
    c1.metric("RF Mean Accuracy",  f"{scores_rf.mean()*100:.2f}%",  f"±{scores_rf.std()*100:.2f}%")
    c2.metric("KNN Mean Accuracy", f"{scores_knn.mean()*100:.2f}%", f"±{scores_knn.std()*100:.2f}%")
    c3.metric("P-Value", f"{p_value:.4f}", "Signifikan ✓" if p_value < 0.05 else "Tidak Signifikan")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-label">Distribusi Skor per Fold</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(5); w = 0.3
        ax.bar(x-w/2, scores_rf*100,  width=w, color=ACCENT, label='Random Forest', edgecolor='none')
        ax.bar(x+w/2, scores_knn*100, width=w, color=MID_C,  label='KNN',           edgecolor='none')
        ax.set_xticks(x); ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
        ax.set_ylabel('Akurasi (%)')
        ax.set_ylim(min(scores_knn.min(),scores_rf.min())*100-2,
                    max(scores_rf.max(),scores_knn.max())*100+2)
        ax.set_title('Akurasi per Fold — RF vs KNN', color='#e8e8f0', fontweight='bold')
        ax.legend(fontsize=9); ax.grid(axis='y')
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-label">Kesimpulan Uji Statistik</div>', unsafe_allow_html=True)
        sig    = p_value < 0.05
        c_sig  = LOW_C if sig else MID_C
        status = "SIGNIFIKAN" if sig else "TIDAK SIGNIFIKAN"
        interp = ("Perbedaan performa RF dan KNN terbukti signifikan secara statistik (p < 0.05). "
                  "Random Forest benar-benar lebih baik, bukan karena kebetulan."
                  if sig else
                  "Perbedaan performa kedua model tidak signifikan secara statistik (p ≥ 0.05). "
                  "Perbedaan akurasi belum terbukti konsisten.")
        st.markdown(f"""
        <div style='background:#111120;border:1px solid rgba(255,255,255,0.07);
        border-radius:16px;padding:1.5rem;height:220px;
        display:flex;flex-direction:column;justify-content:center'>
            <div style='font-family:Syne,sans-serif;font-size:0.68rem;font-weight:700;
            text-transform:uppercase;letter-spacing:2px;color:#6b6b88;margin-bottom:0.8rem'>Kesimpulan A/B Test</div>
            <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;
            color:{c_sig};margin-bottom:0.8rem'>{status}</div>
            <div style='font-size:0.82rem;color:#9090a8;line-height:1.6'>{interp}</div>
            <div style='margin-top:0.8rem;font-size:0.78rem;color:#4a4a6a'>
            t-stat = {t_stat:.4f} · p-value = {p_value:.4f} · α = 0.05</div>
        </div>""", unsafe_allow_html=True)