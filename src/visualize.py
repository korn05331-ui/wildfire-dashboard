"""
visualize.py
============
ฟังก์ชันสำหรับสร้างกราฟและแผนที่ ใช้ทั้งใน notebook (EDA) และในแดชบอร์ด

มี 2 ประเภท:
1. matplotlib/seaborn - สำหรับไฟล์รายงาน (.png)
2. plotly/folium      - สำหรับแดชบอร์ด interactive
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _setup_thai_font():
    """
    ตั้งฟอนต์ไทยให้ matplotlib
    - ลองหาฟอนต์ไทยที่มีอยู่ในระบบ (Tahoma, TH Sarabun, Leelawadee)
    - ถ้าไม่มี ใช้ DejaVu Sans (ภาษาไทยอาจไม่ขึ้น แต่กราฟยังสร้างได้)
    - ใน Colab: แนะนำให้รัน setup_colab_thai_font() ใน notebook
    """
    import matplotlib.font_manager as fm
    candidates = ["Leelawadee UI", "Tahoma", "TH Sarabun New",
                  "Cordia New", "Sukhumvit Set"]
    available = set(f.name for f in fm.fontManager.ttflist)
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            return font
    plt.rcParams["font.family"] = "DejaVu Sans"
    return None


plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")
_setup_thai_font()


def setup_colab_thai_font():
    """
    ใช้ใน Google Colab เพื่อติดตั้งฟอนต์ไทย (TH Sarabun New) และตั้งเป็นค่าเริ่มต้น
    เรียกใช้ฟังก์ชันนี้ที่จุดเริ่มต้นของ notebook ใน Colab
    """
    import subprocess
    import os
    import matplotlib as mpl

    # ติดตั้งฟอนต์ไทย
    subprocess.run(["apt-get", "install", "-y", "--no-install-recommends",
                    "fonts-thai-tlwg"], check=False)
    # ล้าง cache ฟอนต์ของ matplotlib
    cache_dir = mpl.get_cachedir()
    for f in os.listdir(cache_dir):
        if "font" in f.lower():
            try:
                os.remove(os.path.join(cache_dir, f))
            except OSError:
                pass
    # โหลดฟอนต์ใหม่
    mpl.font_manager._load_fontmanager(try_read_cache=False)
    plt.rcParams["font.family"] = "Loma"   # ฟอนต์ TLWG Loma (คล้าย Tahoma)
    plt.rcParams["axes.unicode_minus"] = False
    print("✅ ติดตั้งและตั้งฟอนต์ไทยเรียบร้อย (Loma)")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import config


# ============================================================
# ส่วนที่ 1: กราฟด้วย matplotlib (สำหรับรายงาน)
# ============================================================
def plot_fires_by_month(df: pd.DataFrame, save_path=None):
    """กราฟจำนวนไฟป่าตามเดือน (เห็นฤดูกาลชัด)"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fires = df.groupby("month")["fire"].sum()
    all_days = df.groupby("month").size()
    rate = (fires / all_days * 100).round(2)

    bars = ax.bar(rate.index, rate.values, color="#e74c3c", alpha=0.85)
    ax.set_xlabel("เดือน")
    ax.set_ylabel("อัตราการเกิดไฟป่า (%)")
    ax.set_title("อัตราการเกิดไฟป่ารายเดือน (รวมทุกปี)", fontweight="bold")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(
        ["ม.ค.","ก.พ.","มี.ค.","เม.ย.","พ.ค.","มิ.ย.",
         "ก.ค.","ส.ค.","ก.ย.","ต.ค.","พ.ย.","ธ.ค."]
    )
    for bar, v in zip(bars, rate.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.1f}%",
                ha="center", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_weather_compare_fire(df: pd.DataFrame, save_path=None):
    """
    เปรียบเทียบค่าอากาศระหว่างวันที่ "เกิดไฟ" กับ "ไม่เกิดไฟ"
    เป็น boxplot 4 ช่อง (อุณหภูมิ/ความชื้น/ฝน 7 วัน/ความเร็วลม)
    """
    vars_to_plot = [
        ("temperature", "อุณหภูมิ (°C)"),
        ("humidity",    "ความชื้น (%)"),
        ("rain_7d",     "ฝนสะสม 7 วัน (มม.)"),
        ("wind_speed",  "ความเร็วลม (กม./ชม.)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    df_plot = df.copy()
    df_plot["สถานะ"] = np.where(df_plot["fire"] == 1, "เกิดไฟ", "ไม่เกิดไฟ")

    for ax, (col, label) in zip(axes, vars_to_plot):
        sns.boxplot(data=df_plot, x="สถานะ", y=col, ax=ax,
                    palette={"เกิดไฟ": "#e74c3c", "ไม่เกิดไฟ": "#3498db"},
                    order=["ไม่เกิดไฟ", "เกิดไฟ"])
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("")
    fig.suptitle("เปรียบเทียบสภาพอากาศ: วันเกิดไฟ vs ไม่เกิดไฟ",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save_path=None):
    """Heatmap ความสัมพันธ์ระหว่างตัวแปรอากาศกับการเกิดไฟ"""
    cols = config.WEATHER_VARS + [config.TARGET_COL]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, square=True, linewidths=0.5)
    ax.set_title("ความสัมพันธ์ระหว่างตัวแปร (Correlation)",
                 fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, save_path=None):
    """กราฟแท่งแสดงความสำคัญของแต่ละปัจจัย"""
    d = importance_df.copy().sort_values("importance")
    # แปลชื่อ feature เป็นไทยให้รายงานสวย
    th_names = {
        "temperature": "อุณหภูมิ",
        "humidity":    "ความชื้น",
        "rain":        "ฝนวันนี้",
        "wind_speed":  "ความเร็วลม",
        "rain_7d":     "ฝนสะสม 7 วัน",
        "rain_14d":    "ฝนสะสม 14 วัน",
    }
    d["feature_th"] = d["feature"].map(th_names).fillna(d["feature"])

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(d["feature_th"], d["importance"], color="#2980b9")
    for bar, v in zip(bars, d["importance"]):
        ax.text(v + d["importance"].max() * 0.01, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel(f"ความสำคัญ ({d['method'].iloc[0]})")
    ax.set_title("ลำดับความสำคัญของปัจจัยต่อการเกิดไฟป่า",
                 fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ============================================================
# ส่วนที่ 2: กราฟ interactive ด้วย Plotly (สำหรับแดชบอร์ด)
# ============================================================
def plotly_fires_over_time(df: pd.DataFrame):
    """กราฟเส้นจำนวน hotspot หรือไฟรายวัน/รายเดือน (interactive)"""
    import plotly.express as px

    monthly = (df.groupby([df["date"].dt.to_period("M"), "province"])
                 .agg(hotspots=("hotspots", "sum"),
                      fires=("fire", "sum"))
                 .reset_index())
    monthly["date"] = monthly["date"].dt.to_timestamp()

    fig = px.line(monthly, x="date", y="hotspots", color="province",
                  markers=True,
                  title="จำนวนจุดความร้อนรายเดือน แยกตามจังหวัด",
                  labels={"date": "เดือน", "hotspots": "จำนวน hotspot",
                          "province": "จังหวัด"})
    fig.update_layout(height=400, hovermode="x unified")
    return fig


def plotly_weather_vs_fire(df: pd.DataFrame, var: str, var_label_th: str):
    """กราฟกระจาย (scatter) เปรียบเทียบตัวแปรกับสถานะไฟ"""
    import plotly.express as px
    d = df.sample(min(len(df), 3000), random_state=config.RANDOM_STATE).copy()
    d["สถานะ"] = np.where(d["fire"] == 1, "เกิดไฟ", "ไม่เกิดไฟ")
    fig = px.strip(d, x=var, y="province", color="สถานะ",
                   color_discrete_map={"เกิดไฟ": "#e74c3c", "ไม่เกิดไฟ": "#3498db"},
                   opacity=0.5,
                   title=f"{var_label_th} แยกตามสถานะไฟป่า",
                   labels={var: var_label_th, "province": "จังหวัด"})
    fig.update_layout(height=350)
    return fig


# ============================================================
# ส่วนที่ 3: แผนที่ด้วย Folium (interactive)
# ============================================================
def make_risk_map(df: pd.DataFrame, model, date=None):
    """
    สร้างแผนที่แสดงระดับความเสี่ยงของแต่ละ cell
    - ถ้า date=None จะใช้ค่าเฉลี่ยจากข้อมูลล่าสุดของ cell นั้น
    - ใช้สีเขียว/เหลือง/ส้ม/แดง ตามระดับความเสี่ยง
    """
    import folium

    # ถ้าไม่ระบุ date ใช้วันล่าสุดในข้อมูล
    if date is None:
        date = df["date"].max()

    # กรองเฉพาะวันที่ที่เลือก
    day_data = df[df["date"] == date].copy()
    if len(day_data) == 0:
        # ถ้าไม่มีข้อมูลวันนั้น ใช้ค่าเฉลี่ย cell นั้น
        day_data = df.groupby("cell_id").first().reset_index()

    X = day_data[config.FEATURE_COLS]
    day_data["prob"] = model.predict_proba(X)[:, 1]

    # จุดกลางแผนที่
    center = [day_data["lat"].mean(), day_data["lon"].mean()]
    m = folium.Map(location=center, zoom_start=8, tiles="OpenStreetMap")

    for _, row in day_data.iterrows():
        level = config.get_risk_level(row["prob"])
        popup_text = (
            f"<b>{row['province']}</b> | cell {row['cell_id']}<br>"
            f"วันที่: {row['date'].strftime('%Y-%m-%d')}<br>"
            f"อุณหภูมิ: {row['temperature']}°C<br>"
            f"ความชื้น: {row['humidity']}%<br>"
            f"ฝน 7 วัน: {row['rain_7d']} มม.<br>"
            f"ลม: {row['wind_speed']} กม./ชม.<br>"
            f"<b>ความเสี่ยง: {row['prob']*100:.1f}% ({level['name']})</b>"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=14,
            color=level["color"],
            fill=True,
            fill_color=level["color"],
            fill_opacity=0.75,
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=f"{level['name']} ({row['prob']*100:.0f}%)",
        ).add_to(m)

    # เพิ่ม legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px; border-radius: 6px;
                box-shadow: 0 0 8px rgba(0,0,0,0.2); font-size: 13px;">
      <b>ระดับความเสี่ยง</b><br>
    """
    for level in config.RISK_LEVELS:
        legend_html += (
            f'<i style="background:{level["color"]}; width:12px; height:12px;'
            f'display:inline-block; margin-right:6px;"></i>'
            f'{level["name"]}<br>'
        )
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ============================================================
if __name__ == "__main__":
    from synthetic_data import load_dataset
    from features import build_features
    import os

    df = load_dataset()
    df = build_features(df)

    # ทดสอบสร้างกราฟ
    out_dir = config.PROJECT_ROOT / "notebooks" / "figs"
    out_dir.mkdir(exist_ok=True)

    print("กำลังสร้างกราฟตัวอย่าง...")
    plot_fires_by_month(df, save_path=out_dir / "fires_by_month.png")
    plot_weather_compare_fire(df, save_path=out_dir / "weather_compare.png")
    plot_correlation_heatmap(df, save_path=out_dir / "correlation.png")
    print(f"✅ บันทึกกราฟที่: {out_dir}")
