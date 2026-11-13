import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_auc_score
import joblib

# =============================================================================
# KONFIGURASI
# =============================================================================
st.set_page_config(page_title="Deposit Radar", page_icon="🏦", layout="wide")

MODEL_PATH = "model_gboost_v1.sav"
TARGET_COL = "deposit"  # Asumsikan nama kolom target
ID_COL = "id"           # Opsional jika ada
RANDOM_STATE = 42

# Fitur berdasarkan informasi ColumnTransformer dari log:
# ['age', 'job', 'balance', 'housing', 'loan', 'contact', 'month', 'campaign', 'pdays', 'poutcome']

NUM_COLS = [
    "age", "balance", "campaign", "pdays"
]
CAT_COLS = [
    "job", "housing", "loan", "contact", "month", "poutcome"
]
FEATURE_COLS = NUM_COLS + CAT_COLS

TIER_LABELS = [
    "Tier 1 (Sangat Tinggi)",
    "Tier 2 (Tinggi)",
    "Tier 3 (Rendah - Hold)",
]
TIER1_CUTOFF = 0.75

FEATURE_LABELS = {
    "age": "Usia",
    "job": "Pekerjaan",
    "balance": "Saldo Rata-rata Tahunan",
    "housing": "KPR (Housing Loan)",
    "loan": "Pinjaman Pribadi",
    "contact": "Kanal Komunikasi",
    "month": "Bulan Kontak Terakhir",
    "campaign": "Jumlah Kontak (Kampanye Ini)",
    "pdays": "Jeda Hari (Kampanye Sebelumnya)",
    "poutcome": "Hasil Kampanye Sebelumnya"
}

def label(col):
    return FEATURE_LABELS.get(col, col)

# Warna tema "Ivory & Emerald" (warna dasar widget ada di .streamlit/config.toml)
C_TIER1 = "#0E7C5A"
C_TIER2 = "#6BBF98"
C_TIER3 = "#D8D4C8"
C_THRESHOLD = "#C2410C"
C_MUTED = "#5C6A63"

# =============================================================================
# MODEL PLACEHOLDER
# =============================================================================
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None
    return None

def predict_proba(df):
    model = load_model()
    if model is None:
        # Fallback dummy prediction jika model gagal dimuat
        return np.random.uniform(0.1, 0.9, size=len(df))

    # Model sklearn Pipeline (ColumnTransformer + GradientBoosting)
    # Model diharapkan bisa langsung menerima DataFrame dengan nama kolom yang sesuai
    X = df[FEATURE_COLS].copy()

    # Menangani pdays (jika ada missing value indikator di pipeline)
    if "pdays" in X.columns:
        X["pdays"] = pd.to_numeric(X["pdays"], errors="coerce")

    try:
        return np.asarray(model.predict_proba(X))[:, 1]
    except Exception as e:
        st.error(f"Error saat prediksi: {e}")
        return np.random.uniform(0.1, 0.9, size=len(df))

def is_demo_mode():
    return load_model() is None

# =============================================================================
# DATA & HELPERS
# =============================================================================
def validate_columns(df):
    return [c for c in FEATURE_COLS if c not in df.columns]

def coerce_types(df):
    df = df.copy()
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT_COLS:
        df[c] = df[c].astype("object").where(df[c].notna(), np.nan)
    return df

def risk_label(p, threshold):
    return np.where(p >= threshold, "Deposit (1)", "Tidak Deposit (0)")

def html(markup):
    st.markdown(markup, unsafe_allow_html=True)

def kpi_card(lbl, value, sub, color="var(--dr-text)"):
    return (
        f'<div class="dr-kpi"><div class="dr-lbl">{lbl}</div>'
        f'<div class="dr-num" style="color:{color}">{value}</div>'
        f'<div class="dr-sub">{sub}</div></div>'
    )

def score_histogram(proba, threshold):
    """Histogram skor, batang diwarnai per tier (bin 0.05, ditumpuk di bin batas)."""
    fig = go.Figure()
    bins = dict(start=0, end=1.0001, size=0.05)
    groups = [
        (proba < threshold, C_TIER3, TIER_LABELS[2]),
        ((proba >= threshold) & (proba < TIER1_CUTOFF), C_TIER2, TIER_LABELS[1]),
        (proba >= TIER1_CUTOFF, C_TIER1, TIER_LABELS[0]),
    ]
    for mask, color, name in groups:
        fig.add_trace(go.Histogram(
            x=proba[mask], xbins=bins, marker=dict(color=color, line=dict(width=0)),
            name=name, hovertemplate="Skor %{x}<br>%{y} nasabah<extra>" + name + "</extra>",
        ))
    fig.add_vline(
        x=threshold, line_dash="dash", line_color=C_THRESHOLD, line_width=2,
        annotation_text=f"{threshold:.2f}", annotation_position="top right",
        annotation_font=dict(color=C_THRESHOLD, family="JetBrains Mono", size=12),
    )
    fig.update_layout(
        barmode="stack", bargap=0.12, showlegend=False, height=270,
        margin=dict(l=36, r=8, t=24, b=32),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=C_MUTED, size=12),
        xaxis=dict(range=[0, 1], dtick=0.25, showgrid=False, zeroline=False,
                   tickfont=dict(family="JetBrains Mono")),
        yaxis=dict(showgrid=True, gridcolor="#EEEBE3", zeroline=False, title=None),
    )
    return fig

# =============================================================================
# GAYA HALAMAN
# =============================================================================
html("""
<style>
:root{
  --dr-surface:#FFFFFF; --dr-line:rgba(21,32,27,.10); --dr-text:#15201B; --dr-muted:#5C6A63;
  --dr-faint:#8A958F; --dr-t1:#0E7C5A; --dr-t1-soft:rgba(14,124,90,.12); --dr-t2:#6BBF98;
  --dr-t2-soft:rgba(107,191,152,.16); --dr-t2-text:#2F7A57; --dr-t3:#D8D4C8; --dr-ok:#16A34A;
  --dr-shadow:0 1px 2px rgba(21,32,27,.04),0 8px 24px -12px rgba(21,32,27,.12);
}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1120px;padding-top:2.2rem;padding-bottom:3rem}

/* kartu = st.container(key="card_...") */
[class*="st-key-card_"]{background:var(--dr-surface);border:1px solid var(--dr-line);
  border-radius:18px;padding:22px 22px 18px;box-shadow:var(--dr-shadow)}
.st-key-card_upload,.st-key-card_threshold{min-height:248px}

/* header */
.dr-nav{display:flex;align-items:center;gap:12px;margin-bottom:22px}
.dr-logo{width:38px;height:38px;border-radius:11px;display:grid;place-items:center;background:var(--dr-t1);color:#fff}
.dr-logo svg{width:20px;height:20px;fill:none;stroke:currentColor;stroke-width:2.2}
.dr-nav h1{font-size:23px!important;line-height:1.2!important;padding:0!important;margin:0!important;letter-spacing:-.01em}
.dr-nav p{color:var(--dr-muted);font-size:13px;margin:0}
.dr-sp{flex:1}
.dr-chip{display:inline-flex;align-items:center;gap:8px;padding:6px 13px;border-radius:999px;
  border:1px solid var(--dr-line);background:var(--dr-surface);font-size:13px;color:var(--dr-muted)}
.dr-dot{width:8px;height:8px;border-radius:50%;background:var(--dr-ok)}
.dr-chip.warn{color:#9A3412;border-color:rgba(194,65,12,.3);background:#FFF7ED}
.dr-chip.warn .dr-dot{background:#EA580C}

.dr-lbl{color:var(--dr-muted);font-size:13px;font-weight:500}
.dr-hint{color:var(--dr-faint);font-size:12.5px;margin-top:2px}
.dr-hint code{font-size:11.5px}

/* upload */
[data-testid="stFileUploaderDropzone"]{background:var(--dr-t2-soft);border:1.5px dashed var(--dr-t2);border-radius:14px}

/* threshold */
.dr-th{display:flex;justify-content:space-between;align-items:baseline}
.dr-thv{font:700 30px "Space Grotesk",sans-serif;color:var(--dr-t1);line-height:1}
.dr-scale{display:flex;gap:3px;height:8px;border-radius:99px;overflow:hidden;margin:2px 0 10px}
[data-testid="stSliderThumbValue"]{display:none}
.dr-scale i{display:block;height:100%}

/* KPI */
.dr-kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:8px 0 12px}
.dr-kpi{background:var(--dr-surface);border:1px solid var(--dr-line);border-radius:18px;padding:20px 22px;box-shadow:var(--dr-shadow)}
.dr-num{font:700 34px/1.15 "Space Grotesk",sans-serif;letter-spacing:-.02em;margin-top:4px}
.dr-sub{font-size:12.5px;color:var(--dr-muted)}
@media (max-width:760px){.dr-kpis{grid-template-columns:repeat(2,1fr)}}

/* judul kartu */
.dr-hd{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px}
.dr-hd h3{font-size:16px!important;font-weight:600!important;padding:0!important;margin:0!important}
.dr-hd small{color:var(--dr-muted);font-size:12.5px}

/* segmen */
.dr-stack{display:flex;height:14px;border-radius:99px;overflow:hidden;gap:3px;margin:14px 0 18px}
.dr-tier{display:flex;align-items:center;gap:12px;padding:12px 0;border-top:1px solid var(--dr-line)}
.dr-tier i{width:10px;height:10px;border-radius:3px;flex:none}
.dr-tier div{flex:1}
.dr-tier b{display:block;font-weight:600;font-size:14px;color:var(--dr-text)}
.dr-tier small{color:var(--dr-muted);font-size:12px}
.dr-tier .n{font:700 20px "Space Grotesk",sans-serif;color:var(--dr-text)}

/* kosong */
.dr-empty{text-align:center;padding:34px 10px}
.dr-empty b{display:block;font:600 17px "Space Grotesk",sans-serif;margin-bottom:4px}
.dr-empty span{color:var(--dr-muted);font-size:14px}
</style>
""")

# =============================================================================
# UI: HEADER
# =============================================================================
if is_demo_mode():
    status_chip = '<span class="dr-chip warn"><span class="dr-dot"></span>Mode demo: model gagal dimuat, skor acak</span>'
else:
    status_chip = '<span class="dr-chip"><span class="dr-dot"></span>Model Gradient Boosting aktif</span>'

html(
    '<div class="dr-nav">'
    '<div class="dr-logo"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/></svg></div>'
    '<div><h1>Deposit Radar</h1><p>Prediksi nasabah yang akan membuka deposito · batch scoring</p></div>'
    f'<div class="dr-sp"></div>{status_chip}</div>'
)

# =============================================================================
# UI: UPLOAD + THRESHOLD
# =============================================================================
col_up, col_th = st.columns(2, gap="medium")

with col_up:
    with st.container(key="card_upload"):
        html('<div class="dr-lbl">Data nasabah</div>')
        batch_file = st.file_uploader(
            "Upload CSV", type="csv", key="batch_upload", label_visibility="collapsed"
        )
        template = pd.DataFrame(columns=[ID_COL] + FEATURE_COLS)
        st.download_button(
            "Unduh template CSV", template.to_csv(index=False).encode(),
            "template_deposit.csv", "text/csv",
            type="tertiary", icon=":material/description:",
        )
        html(
            f'<div class="dr-hint">Kolom wajib: <code>{", ".join(FEATURE_COLS)}</code>. '
            f'Opsional: <code>{ID_COL}</code>, <code>{TARGET_COL}</code> (untuk evaluasi).</div>'
        )

with col_th:
    with st.container(key="card_threshold"):
        thr_now = st.session_state.get("threshold", 0.45)
        html(
            '<div class="dr-th"><span class="dr-lbl">Threshold keputusan</span>'
            f'<span class="dr-thv">{thr_now:.2f}</span></div>'
        )
        threshold = st.slider(
            "Threshold keputusan", 0.01, 0.99, 0.45, 0.01, key="threshold",
            label_visibility="collapsed",
            help="Probabilitas ≥ threshold → Diprediksi Deposit (Hot Lead)",
        )
        t2_width = max(TIER1_CUTOFF - threshold, 0)
        html(
            '<div class="dr-scale">'
            f'<i style="flex:{threshold};background:var(--dr-t3)"></i>'
            f'<i style="flex:{t2_width};background:var(--dr-t2)"></i>'
            f'<i style="flex:{1 - TIER1_CUTOFF};background:var(--dr-t1)"></i></div>'
            f'<div class="dr-hint">Skor ≥ {threshold:.2f} dianggap hot lead · '
            f'skor ≥ {TIER1_CUTOFF:.2f} masuk Tier 1.</div>'
        )

# =============================================================================
# UI: HASIL PREDIKSI
# =============================================================================
if batch_file is None:
    with st.container(key="card_empty"):
        html(
            '<div class="dr-empty"><b>Belum ada data</b>'
            '<span>Upload file CSV nasabah untuk melihat skor dan daftar prioritas telemarketing.</span></div>'
        )
    st.stop()

batch = pd.read_csv(batch_file)

# Huruf kecilkan semua nama kolom agar cocok dengan fitur yang diharapkan
batch.columns = [c.lower() for c in batch.columns]

missing = validate_columns(batch)
if missing:
    st.error(f"Kolom berikut tidak ditemukan: {missing}")
    st.stop()

batch = coerce_types(batch)
n_bad_rows = batch[NUM_COLS].isna().any(axis=1).sum()
if n_bad_rows:
    st.warning(f"{n_bad_rows} baris punya nilai numerik kosong/tidak valid. (Pipeline model harus memiliki imputer untuk ini).")

with st.spinner("Memprediksi probabilitas..."):
    proba = predict_proba(batch)

result = batch.copy()
result["probabilitas_deposit"] = proba
result["prediksi_deposit"] = (proba >= threshold).astype(int)
result["label_prediksi"] = risk_label(proba, threshold)

# Segmentasi prioritas
result['prioritas_telemarketing'] = np.select(
    [
        result['probabilitas_deposit'] >= TIER1_CUTOFF,
        result['probabilitas_deposit'] >= threshold,
        result['probabilitas_deposit'] < threshold
    ],
    TIER_LABELS,
    default=TIER_LABELS[2]
)

n_total = len(result)
n_hot = int(result["prediksi_deposit"].sum())
tier_counts = result["prioritas_telemarketing"].value_counts().reindex(TIER_LABELS, fill_value=0)
n_t1, n_t2, n_t3 = (int(tier_counts[t]) for t in TIER_LABELS)

# --- KPI ---
html(
    '<div class="dr-kpis">'
    + kpi_card("Total prospek", f"{n_total:,}", "baris diskor")
    + kpi_card("Hot leads", f"{n_hot:,}", f"{n_hot / n_total:.1%} dari total", "var(--dr-t2-text)")
    + kpi_card("Tier 1", f"{n_t1:,}", "telepon lebih dulu", "var(--dr-t1)")
    + kpi_card("Rata-rata skor", f"{proba.mean():.3f}", "probabilitas deposit")
    + '</div>'
)

# --- Distribusi + segmen ---
col_hist, col_seg = st.columns([1.7, 1], gap="medium")

with col_hist:
    with st.container(key="card_hist"):
        html('<div class="dr-hd"><h3>Distribusi skor</h3><small>warna = tier</small></div>')
        st.plotly_chart(
            score_histogram(proba, threshold), width="stretch", theme=None,
            config={"displayModeBar": False},
        )

with col_seg:
    with st.container(key="card_segmen"):
        rows = [
            (n_t1, "var(--dr-t1)", "Tier 1", f"skor ≥ {TIER1_CUTOFF:.2f} · telepon dulu"),
            (n_t2, "var(--dr-t2)", "Tier 2", f"{threshold:.2f} – {TIER1_CUTOFF:.2f} · follow-up"),
            (n_t3, "var(--dr-t3)", "Tier 3", f"&lt; {threshold:.2f} · tahan dulu"),
        ]
        stack = "".join(
            f'<i style="flex:{n};background:{c}"></i>' for n, c, _, _ in rows if n
        )
        tiers = "".join(
            f'<div class="dr-tier"><i style="background:{c}"></i>'
            f'<div><b>{name}</b><small>{desc}</small></div><span class="n">{n:,}</span></div>'
            for n, c, name, desc in rows
        )
        html(f'<div class="dr-hd"><h3>Segmen prioritas</h3></div><div class="dr-stack">{stack}</div>{tiers}')

# --- Tabel + evaluasi ---
with st.container(key="card_tabel"):
    tab_daftar, tab_eval = st.tabs(["Daftar prioritas", "Evaluasi model"])

    with tab_daftar:
        ranked = result.sort_values("probabilitas_deposit", ascending=False)
        punya_id = ID_COL in ranked.columns

        f1, f2 = st.columns([3, 2])
        with f1:
            tier_pilihan = st.pills(
                "Filter tier", TIER_LABELS, selection_mode="multi",
                default=TIER_LABELS, key="filter_tier",
            )
        with f2:
            cari_id = st.text_input(
                "Cari ID", placeholder="contoh: 716, 302",
                disabled=not punya_id, key="cari_id",
                help="Pisahkan beberapa ID dengan koma atau spasi." if punya_id
                else f"File tidak punya kolom `{ID_COL}`.",
            )

        # Filter: tier dulu, lalu ID (cocok persis, bisa lebih dari satu)
        view = ranked[ranked["prioritas_telemarketing"].isin(tier_pilihan)]
        id_dicari = [s for s in re.split(r"[,\s]+", cari_id.strip()) if s] if punya_id else []
        if id_dicari:
            view = view[view[ID_COL].astype(str).isin(id_dicari)]

        st.caption(f"Menampilkan **{len(view):,}** dari **{len(ranked):,}** nasabah · urut skor tertinggi")

        if view.empty:
            st.info("Tidak ada nasabah yang cocok. Pilih minimal satu tier atau periksa ID yang dicari.")
        else:
            depan = [c for c in [ID_COL, "prioritas_telemarketing", "probabilitas_deposit"] if c in view.columns]
            urutan = depan + [c for c in view.columns if c not in depan]
            kolom = {c: st.column_config.Column(label(c)) for c in FEATURE_COLS}
            kolom.update({
                ID_COL: st.column_config.TextColumn("ID"),
                "prioritas_telemarketing": st.column_config.TextColumn("Prioritas"),
                "probabilitas_deposit": st.column_config.ProgressColumn(
                    "Skor deposit", format="%.2f", min_value=0.0, max_value=1.0
                ),
                "prediksi_deposit": st.column_config.NumberColumn("Prediksi (0/1)"),
                "label_prediksi": st.column_config.TextColumn("Label prediksi"),
            })
            st.dataframe(
                view, width="stretch", height=420, hide_index=True,
                column_order=urutan, column_config=kolom,
            )

        # Ekspor CSV sesuai filter yang aktif
        if set(tier_pilihan) == set(TIER_LABELS) and not id_dicari:
            nama_file = "telemarketing_priority_list.csv"
        else:
            bagian = [f"tier{i + 1}" for i, t in enumerate(TIER_LABELS) if t in tier_pilihan]
            if id_dicari:
                bagian.append("cari_id")
            nama_file = "telemarketing_" + "_".join(bagian) + ".csv"

        b1, b2, _ = st.columns([1.4, 1, 1.6])
        with b1:
            st.download_button(
                f"Unduh daftar prioritas ({len(view):,} baris)",
                view.to_csv(index=False).encode(), nama_file, "text/csv",
                type="primary", icon=":material/download:",
                disabled=view.empty, width="stretch",
            )
        with b2:
            tier1_only = ranked[ranked["prioritas_telemarketing"] == TIER_LABELS[0]]
            st.download_button(
                f"Tier 1 saja ({len(tier1_only):,})",
                tier1_only.to_csv(index=False).encode(), "telemarketing_tier1.csv", "text/csv",
                type="secondary", disabled=tier1_only.empty, width="stretch",
            )

    with tab_eval:
        if TARGET_COL in batch.columns and batch[TARGET_COL].nunique() == 2:
            y = batch[TARGET_COL].astype(int)
            tn, fp, fn, tp = confusion_matrix(y, result["prediksi_deposit"], labels=[0, 1]).ravel()
            html(
                '<div class="dr-kpis">'
                + kpi_card("ROC-AUC", f"{roc_auc_score(y, proba):.3f}", "pemisahan kelas")
                + kpi_card("Precision", f"{tp / (tp + fp):.3f}" if (tp + fp) else "-", "hot lead yang benar deposit", "var(--dr-t1)")
                + kpi_card("Recall", f"{tp / (tp + fn):.3f}" if (tp + fn) else "-", "deposit yang tertangkap", "var(--dr-t2-text)")
                + kpi_card("Accuracy", f"{(tp + tn) / len(y):.3f}", f"pada threshold {threshold:.2f}")
                + '</div>'
            )
            st.caption(f"Confusion matrix: TP {tp:,} · FP {fp:,} · FN {fn:,} · TN {tn:,}")
        else:
            st.info(
                f"Evaluasi butuh kolom `{TARGET_COL}` berisi 0/1 di file CSV. "
                "File yang di-upload tidak punya kolom ini, jadi hanya prediksi yang ditampilkan."
            )
