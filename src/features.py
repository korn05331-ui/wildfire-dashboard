"""
features.py
===========
สร้างตัวแปร (feature engineering) สำหรับโมเดล

ตัวแปรที่สร้างเพิ่ม:
- rain_7d  : ฝนสะสมย้อนหลัง 7 วัน  (แห้งแล้ง→เชื้อเพลิงแห้ง→ไฟลุกง่าย)
- rain_14d : ฝนสะสมย้อนหลัง 14 วัน

หลักการ: "ฝนวันนี้" ไม่พิการการเกิดไฟมากเท่า "ฝนสะสม"
         ถ้าฝนไม่ตกมา 14 วัน ใบไม้/หญ้าจะแห้ง = ไฟลุกง่ายมาก
"""

import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import config


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    เพิ่มคอลัมน์ฝนสะสม 7 และ 14 วัน
    **สำคัญ:** ต้องคำนวณแยกตาม cell_id (ทำ rolling เฉพาะใน cell เดียวกัน)
    ถ้าไม่ group ค่าจะ "หลุด" ไป cell อื่น
    """
    df = df.sort_values(["cell_id", "date"]).copy()

    # ฝนสะสม 7 วัน = ผลรวมฝนวันนี้ถึง 6 วันก่อน
    df["rain_7d"] = (
        df.groupby("cell_id")["rain"]
          .rolling(window=7, min_periods=1)
          .sum()
          .reset_index(level=0, drop=True)
    )

    # ฝนสะสม 14 วัน
    df["rain_14d"] = (
        df.groupby("cell_id")["rain"]
          .rolling(window=14, min_periods=1)
          .sum()
          .reset_index(level=0, drop=True)
    )

    df["rain_7d"] = df["rain_7d"].round(1)
    df["rain_14d"] = df["rain_14d"].round(1)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    เพิ่มตัวแปรเวลา (เดือน/วันในปี) ใช้ตอนวิเคราะห์/แดชบอร์ด
    - month: เดือน 1-12
    - day_of_year: ลำดับวันในปี 1-365
    - season: ชื่อฤดู (ใช้แดชบอร์ด)
    """
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    # แบ่งฤดูแบบไทย (ใช้แดชบอร์ด)
    def _season(m):
        if m in (3, 4, 5):        return "ฤดูร้อน"
        if m in (6, 7, 8, 9, 10): return "ฤดูฝน"
        return "ฤดูหนาว"          # 11, 12, 1, 2
    df["season"] = df["month"].apply(_season)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """รวมทุกขั้นตอน feature engineering เป็นฟังก์ชันเดียว"""
    df = add_rolling_features(df)
    df = add_time_features(df)
    return df


def get_feature_matrix(df: pd.DataFrame):
    """
    แยก X (feature) และ y (label) สำหรับเทรนโมเดล
    คืนค่า (X, y) โดย X มีเฉพาะ FEATURE_COLS ที่กำหนดใน config
    """
    X = df[config.FEATURE_COLS].copy()
    y = df[config.TARGET_COL].copy()
    return X, y


# ============================================================
if __name__ == "__main__":
    from synthetic_data import load_dataset

    df = load_dataset()
    print(f"ก่อน: {df.shape}  คอลัมน์ = {list(df.columns)}")
    df = build_features(df)
    print(f"หลัง: {df.shape}  คอลัมน์ = {list(df.columns)}")
    print("\nตัวอย่างฝนสะสม 7 วัน (cell เดียว ช่วงแรก):")
    print(df[["cell_id", "date", "rain", "rain_7d", "rain_14d"]].head(15))
