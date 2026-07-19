"""
app.py — แดชบอร์ด Streamlit ประเมินความเสี่ยงไฟป่า
====================================================
เวอร์ชันสำหรับ Streamlit Community Cloud

โครงสร้างไฟล์ (root):
    app.py              ← ไฟล์นี้ (entry point)
    requirements.txt
    packages.txt        ← ฟอนต์ไทย (apt-get)
    src/                ← โค้ดทั้งหมด
    data/processed/     ← ข้อมูล CSV
    models/             ← โมเดล .joblib

★ Streamlit Cloud จะเรียก app.py นี้โดยอัตโนมัติ
★ ต้องตั้งค่าในแพลตฟอร์ม: Main file path = app.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ทำให้ import src/ ได้จาก root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.config as config
from src.features import build_features
from src.models import load_model


# ============================================================
# ตั้งค่าฟอนต์ไทย (สำคัญ — Cloud มี DejaVu Sans แค่นั้น)
# ============================================================
def setup_thai_font_cloud():
    """
    บน Streamlit Cloud ต้องติดตั้งฟอนต์ไทยผ่าน packages.txt
    (apt-get install fonts-thai-tlwg) ไม่งั้นเป็น □□□

    ฟังก์ชันนี้ล้าง cache ฟอนต์เก่าของ matplotlib แล้วตั้งฟอนต์ไทย
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import glob
    # ล้าง cache เก่า (ถ้าไม่ล้าง matplotlib จะใช้ cache ที่ยังไม่มีฟอนต์ไทย)
    cache_dir = matplotlib.get_cachedir()
    for f in glob.glob(os.path.join(cache_dir, "fontlist*.json")):
        try:
            os.remove(f)
        except OSError:
            pass
    # รีโหลด font manager เพื่อให้เห็นฟอนต์ใหม่
    import matplotlib.font_manager as fm
    fm._load_fontmanager(try_read_cache=False)

    # ลำดับฟอนต์ไทยที่จะลองหา (ติดตั้งผ่าน packages.txt)
    candidates = ["Loma", "Norasi", "Kinnari", "Garuda",
                  "Leelawadee UI", "Tahoma"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            plt.rcParams["axes.unicode_minus"] = False
            return font
    plt.rcParams["font.family"] = "DejaVu Sans"
    return None


# ============================================================
# ตั้งค่าหน้าต่าง
# ============================================================
st.set_page_config(
    page_title="🔥 แดชบอร์ดความเสี่ยงไฟป่า",
    page_icon="🔥",
    layout="wide",
)

setup_thai_font_cloud()


# ============================================================
# โหลดข้อมูลและโมเดล (cache ไว้)
# ============================================================
@st.cache_data
def load_data():
    from src.synthetic_data import load_dataset
    df = load_dataset()
    df = build_features(df)
    return df


@st.cache_resource
def load_best_model():
    best_file = config.MODELS_DIR / "best_model.txt"
    if best_file.exists():
        name = best_file.read_text(encoding="utf-8").strip()
    else:
        name = "rf"
    filename = f"{name}_model.joblib"
    return load_model(filename), name


def load_importance():
    path = config.MODELS_DIR / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ----- โหลด -----
try:
    df = load_data()
    model, model_name = load_best_model()
    importance_df = load_importance()
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    st.stop()


# ============================================================
# แถบด้านข้าง
# ============================================================
st.sidebar.title("🔥 แดชบอร์ดไฟป่า")
st.sidebar.markdown(f"**โมเดล:** {model_name}")

page = st.sidebar.radio(
    "เลือกหน้า",
    ["📊 ภาพรวม", "🔬 วิเคราะห์ปัจจัย", "🎮 ทดลองปรับค่า",
     "🚨 พื้นที่เฝ้าระวัง", "🧠 ข้อมูลโมเดล"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("ตัวกรอง")
province_sel = st.sidebar.multiselect(
    "จังหวัด", options=list(config.PROVINCES.keys()),
    default=list(config.PROVINCES.keys())
)
year_sel = st.sidebar.multiselect(
    "ปี", options=sorted(df["date"].dt.year.unique()),
    default=sorted(df["date"].dt.year.unique())
)

mask = (df["province"].isin(province_sel)) & (df["date"].dt.year.isin(year_sel))
df_filtered = df[mask].copy()


def predict_risk(features_dict):
    """ทำนายความเสี่ยงจาก dict ของ feature"""
    X = pd.DataFrame([features_dict])[config.FEATURE_COLS]
    prob = model.predict_proba(X)[0, 1]
    level = config.get_risk_level(prob)
    return prob, level


# ============================================================
# หน้า 1: ภาพรวม
# ============================================================
if page == "📊 ภาพรวม":
    st.title("📊 ภาพรวมความเสี่ยงไฟป่า")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("จำนวนวัน-พื้นที่", f"{len(df_filtered):,}")
    c2.metric("วันที่เกิดไฟ", f"{int(df_filtered['fire'].sum()):,}")
    c3.metric("อัตราไฟป่า", f"{df_filtered['fire'].mean()*100:.2f}%")
    c4.metric("จุดความร้อนรวม", f"{int(df_filtered['hotspots'].sum()):,}")

    st.markdown("---")
    st.subheader("🗺️ แผนที่ความเสี่ยงไฟป่า")
    st.caption("เลือกวันที่เพื่อดูความเสี่ยงของแต่ละพื้นที่ในวันนั้น")

    available_dates = sorted(df_filtered["date"].dt.strftime("%Y-%m-%d").unique())
    sel_date_str = st.selectbox(
        "เลือกวันที่", options=available_dates,
        index=len(available_dates) - 1,
    )
    sel_date = pd.to_datetime(sel_date_str)

    from src.visualize import make_risk_map
    m = make_risk_map(df_filtered, model, date=sel_date)
    try:
        from streamlit_folium import st_folium
        st_folium(m, width=900, height=500)
    except ImportError:
        st.components.v1.html(m._repr_html_(), height=550)

    st.markdown("---")
    st.subheader("📈 จำนวนไฟป่ารายเดือน")
    import plotly.express as px
    monthly = (df_filtered.groupby([df_filtered["date"].dt.to_period("M"), "province"])
                          .agg(hotspots=("hotspots", "sum"),
                               fires=("fire", "sum"))
                          .reset_index())
    monthly["date"] = monthly["date"].dt.to_timestamp()
    fig = px.bar(monthly, x="date", y="hotspots", color="province",
                 labels={"date": "เดือน", "hotspots": "จำนวน hotspot",
                         "province": "จังหวัด"},
                 title="จำนวนจุดความร้อนรายเดือน")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# หน้า 2: วิเคราะห์ปัจจัย
# ============================================================
elif page == "🔬 วิเคราะห์ปัจจัย":
    st.title("🔬 วิเคราะห์ความสัมพันธ์ปัจจัยอากาศกับไฟป่า")
    st.write("ดูว่าแต่ละตัวแปรอากาศมีความแตกต่างอย่างไรในวันที่เกิดไฟ vs ไม่เกิดไฟ")

    var_options = {
        "temperature": "อุณหภูมิ (°C)",
        "humidity":    "ความชื้น (%)",
        "rain":        "ปริมาณฝนวันนี้ (มม.)",
        "rain_7d":     "ฝนสะสม 7 วัน (มม.)",
        "rain_14d":    "ฝนสะสม 14 วัน (มม.)",
        "wind_speed":  "ความเร็วลม (กม./ชม.)",
    }
    sel_var = st.selectbox("เลือกตัวแปร", list(var_options.keys()),
                            format_func=lambda x: var_options[x])

    import plotly.express as px
    d = df_filtered.copy()
    d["สถานะ"] = np.where(d["fire"] == 1, "เกิดไฟ", "ไม่เกิดไฟ")
    fig = px.box(d, x="สถานะ", y=sel_var, color="สถานะ",
                 color_discrete_map={"เกิดไฟ": "#e74c3c", "ไม่เกิดไฟ": "#3498db"},
                 labels={sel_var: var_options[sel_var]},
                 title=f"{var_options[sel_var]} ในวันที่เกิดไฟ vs ไม่เกิดไฟ")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 ตารางค่าเฉลี่ยตามสถานะ")
    summary = (df_filtered.groupby("fire")[list(var_options.keys())]
                          .mean().round(2).T)
    summary.columns = ["ไม่เกิดไฟ", "เกิดไฟ"]
    st.dataframe(summary, use_container_width=True)


# ============================================================
# หน้า 3: ทดลองปรับค่า
# ============================================================
elif page == "🎮 ทดลองปรับค่า":
    st.title("🎮 ทดลองปรับค่าอากาศ → ดูความเสี่ยง")
    st.write("ปรับค่าสภาพอากาศแล้วดูว่าโมเดลทำนายความเสี่ยงไฟป่าอย่างไร")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("ปรับค่า")
        temp = st.slider("🌡️ อุณหภูมิ (°C)", 10.0, 45.0, 30.0, 0.5)
        humid = st.slider("💧 ความชื้น (%)", 10.0, 100.0, 60.0, 1.0)
        rain_today = st.slider("🌧️ ฝนวันนี้ (มม.)", 0.0, 50.0, 0.0, 0.5)
        wind = st.slider("💨 ความเร็วลม (กม./ชม.)", 0.0, 40.0, 10.0, 0.5)

        st.markdown("**ฝนสะสม** (ลองเปลี่ยนเพื่อดูผล)")
        rain_7d = st.slider("ย้อนหลัง 7 วัน", 0.0, 100.0, 5.0, 1.0)
        rain_14d = st.slider("ย้อนหลัง 14 วัน", 0.0, 200.0, 10.0, 1.0)

    features = {
        "temperature": temp,
        "humidity": humid,
        "rain": rain_today,
        "wind_speed": wind,
        "rain_7d": rain_7d,
        "rain_14d": rain_14d,
    }
    prob, level = predict_risk(features)

    with col2:
        st.subheader("ผลทำนาย")
        st.markdown(
            f"""<div style='background:{level["color"]}; padding:30px;
                border-radius:12px; text-align:center; color:white;'>
                <h1 style='margin:0;'>{prob*100:.1f}%</h1>
                <h2 style='margin:0;'>ระดับ{level["name"]}</h2>
            </div>""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.progress(min(prob, 1.0))
        st.caption(f"ความน่าจะเป็นการเกิดไฟป่า: {prob*100:.2f}%")

        st.markdown("---")
        if prob < 0.20:
            st.success("✅ ความเสี่ยง **ต่ำ** สภาพอากาศยังไม่เอื้อต่อไฟป่า")
        elif prob < 0.40:
            st.info("⚠️ ความเสี่ยง **ปานกลาง** ควรเฝ้าระวังในพื้นที่เปราะบาง")
        elif prob < 0.70:
            st.warning("🚨 ความเสี่ยง **สูง** ควรเตรียมพร้อม + งดเผา")
        else:
            st.error("🔥 ความเสี่ยง **วิกฤต**! ควรออกประกาศเตือนภัยทันที")


# ============================================================
# หน้า 4: พื้นที่เฝ้าระวัง
# ============================================================
elif page == "🚨 พื้นที่เฝ้าระวัง":
    st.title("🚨 พื้นที่เฝ้าระวังไฟป่า")
    st.write("พื้นที่ที่มีความเสี่ยงสูงที่สุดในวันที่เลือก")

    latest_date = df_filtered["date"].max()
    latest = df_filtered[df_filtered["date"] == latest_date].copy()
    X_latest = latest[config.FEATURE_COLS]
    latest["prob"] = model.predict_proba(X_latest)[:, 1]
    latest["level"] = latest["prob"].apply(lambda p: config.get_risk_level(p)["name"])
    latest = latest.sort_values("prob", ascending=False).reset_index(drop=True)

    show_cols = ["province", "cell_id", "temperature", "humidity",
                 "rain_7d", "wind_speed", "prob", "level"]
    rename = {
        "province": "จังหวัด", "cell_id": "พื้นที่",
        "temperature": "อุณหภูมิ (°C)", "humidity": "ความชื้น (%)",
        "rain_7d": "ฝน 7 วัน", "wind_speed": "ลม (กม./ชม.)",
        "prob": "ความเสี่ยง (%)", "level": "ระดับ",
    }
    display = latest[show_cols].copy()
    display["prob"] = (display["prob"] * 100).round(1)
    display = display.rename(columns=rename)
    st.dataframe(display, use_container_width=True, height=400)

    st.caption(f"วันที่แสดง: {latest_date.strftime('%Y-%m-%d')}")


# ============================================================
# หน้า 5: ข้อมูลโมเดล
# ============================================================
elif page == "🧠 ข้อมูลโมเดล":
    st.title("🧠 ข้อมูลโมเดล")
    st.write(f"โมเดลที่ใช้: **{model_name}**")

    st.subheader("🏆 ลำดับความสำคัญของปัจจัย (Feature Importance)")
    if len(importance_df) > 0:
        import plotly.express as px
        th_names = {
            "temperature": "อุณหภูมิ", "humidity": "ความชื้น",
            "rain": "ฝนวันนี้", "wind_speed": "ความเร็วลม",
            "rain_7d": "ฝนสะสม 7 วัน", "rain_14d": "ฝนสะสม 14 วัน",
        }
        d = importance_df.copy()
        d["label"] = d["feature"].map(th_names).fillna(d["feature"])
        d = d.sort_values("importance")
        fig = px.bar(d, x="importance", y="label", orientation="h",
                     labels={"importance": f"ความสำคัญ ({d['method'].iloc[0]})",
                             "label": "ปัจจัย"},
                     title="ลำดับความสำคัญของแต่ละปัจจัยต่อการเกิดไฟป่า",
                     color="importance", color_continuous_scale="Blues")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ยังไม่มีข้อมูล feature importance")

    st.markdown("---")
    st.subheader("📖 คำอธิบายโมเดล")
    st.markdown("""
โครงงานนี้ใช้ 2 โมเดลเปรียบเทียบกัน:

| โมเดล | จุดเด่น | จุดอ่อน |
|-------|--------|--------|
| **Logistic Regression** | อธิบายง่าย (บอกได้ว่าปัจจัยไหนเพิ่ม/ลดความเสี่ยง) | จับความสัมพันธ์ซับซ้อนไม่ได้ |
| **Random Forest** | แม่นยำกว่า จับความสัมพันธ์ซับซ้อนได้ | อธิบายยากกว่า |

การประเมินใช้:
- **ROC-AUC** — ภาพรวมความสามารถแยกไฟ/ไม่ไฟ (≥ 0.5 ดีกว่าสุ่ม)
- **PR-AUC** — เหมาะตอน class imbalance (ไฟน้อยกว่าไม่ไฟมาก)
- **Precision/Recall** — ดูการจับไฟจริง vs การแจ้งเตือนเท็จ
""")
