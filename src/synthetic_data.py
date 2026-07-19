"""
synthetic_data.py
=================
สร้างข้อมูลตัวอย่าง (synthetic) ที่สมจริงสำหรับโครงงานวิเคราะห์ไฟป่า

ทำไมต้องสร้างข้อมูลตัวอย่าง?
- เพื่อให้ทดสอบทั้งระบบ (EDA → โมเดล → แดชบอร์ด) ได้ทันที
  โดยยังไม่ต้องเสียเวลาสมัครและดึงข้อมูลจาก Google Earth Engine
- เมื่อเข้าใจระบบแล้ว ให้สลับเป็นข้อมูลจริงใน gee_extraction.py (เร็ว ๆ นี้)

หลักการสร้างข้อมูลให้ "สมจริง":
1. อุณหภูมิ ขึ้น-ลง ตามฤดูกาล (ร้อนเดือน เม.ย. เย็นเดือน ธ.ค.)
2. ฝน เยอะในฤดูฝน (ม.ค.-เม.ย. แห้งแล้ง, พ.ค.-ต.ค. ฝนตก)
3. ความชื้น สูงในฤดูฝน ต่ำในฤดูแล้ง
4. โอกาสเกิดไฟป่า ขึ้นกับ อุณหภูมิสูง + ความชื้นต่ำ + ฝนน้อย + ลมแรง
   (เพื่อให้โมเดลสามารถหาความสัมพันธ์ได้จริง)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# นำเข้า config (ใช้แบบ relative import เพื่อให้รันได้ทั้งใน/นอก Colab)
import sys
sys.path.append(str(Path(__file__).resolve().parent))
import config


def _seasonal_pattern(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    สร้างค่าฤดูกาลพื้นฐาน (cos/sin) จากวันที่
    เอาไว้คำนวณอุณหภูมิ ฝน ฯลฯ ให้ขึ้น-ลงเป็นรอบปี
    """
    # วันที่ในรอบปี (1-365)
    day_of_year = dates.dayofyear.values
    # ใช้ sin/cos ของมุมวันในปี (1 ปี = 2π)
    angle = 2 * np.pi * (day_of_year - 1) / 365.0
    return pd.DataFrame({
        "date": dates,
        "day_of_year": day_of_year,
        "season_sin": np.sin(angle),   # บวกสุด ~วันที่ 110 (เม.ย.)
        "season_cos": np.cos(angle),
    })


def _generate_weather_one_cell(rng: np.random.Generator,
                                dates: pd.DatetimeIndex,
                                base_temp: float = 28.0) -> pd.DataFrame:
    """
    สร้างข้อมูลอากาศรายวันสำหรับ 1 grid cell
    - base_temp: อุณหภูมิเฉลี่ยประจำ cell (แต่ละพื้นที่ต่างกันเล็กน้อย)
    """
    s = _seasonal_pattern(dates)

    # ---------- อุณหภูมิ (°C) ----------
    # เฉลี่ย 28°C ขึ้น-ลงตามฤดู ช่วงเม.ย.ร้อนสุด ~36°C, ธ.ค.เย็นสุด ~18°C
    temperature = (
        base_temp
        + 6.5 * s["season_sin"]                 # ฤดูกาล
        + rng.normal(0, 1.5, size=len(s))        # noise รายวัน
    )

    # ---------- ความชื้นสัมพัทธ์ (%) ----------
    # ต่ำในฤดูแล้ง (~35%) สูงในฤดูฝน (~80%)
    # สัมพันธ์กลับกับอุณหภูมิ (ยิ่งร้อน ยิ่งแห้ง)
    humidity = (
        75
        - 18 * s["season_sin"]                  # ฤดูกาล (สูงในหนาว-ฝน)
        - 0.8 * (temperature - base_temp)        # ยิ่งร้อนยิ่งแห้ง
        + rng.normal(0, 5, size=len(s))
    )
    humidity = np.clip(humidity, 15, 100)        # จำกัด 15-100%

    # ---------- ปริมาณฝน (มม.) ----------
    # ฝนตกช่วง พ.ค.-ต.ค. (ฤดูฝน) แทบไม่มีเลย ม.ค.-เม.ย.
    # ใช้ฟังก์ชัน "ความน่าจะมีฝน" * "ปริมาณถ้ามี"
    rain_prob = np.clip(0.15 + 0.55 * s["season_cos"] + rng.normal(0, 0.05, len(s)), 0, 0.95)
    has_rain = rng.random(len(s)) < rain_prob
    rain_amount = np.where(
        has_rain,
        rng.gamma(shape=1.5, scale=8.0, size=len(s)),  # ถ้ามีฝน ~Gamma
        0.0,
    )
    rain = np.round(rain_amount, 1)

    # ---------- ความเร็วลม (กม./ชม.) ----------
    # ลมแรงขึ้นในช่วงเปลี่ยนฤดู (~มี.ค.-เม.ย.)
    wind_speed = (
        10
        + 4 * s["season_sin"]
        + rng.normal(0, 2, size=len(s))
    )
    wind_speed = np.clip(wind_speed, 0, 40)

    df = pd.DataFrame({
        "date": dates,
        "temperature": np.round(temperature, 1),
        "humidity": np.round(humidity, 1),
        "rain": rain,
        "wind_speed": np.round(wind_speed, 1),
    })
    return df


def _generate_fire_label(df_weather: pd.DataFrame,
                          rng: np.random.Generator,
                          fire_base_rate: float = 0.04) -> pd.DataFrame:
    """
    สร้าง label 'fire' (0/1) โดยให้ความน่าจะเป็นเกิดไฟป่า
    ขึ้นกับปัจจัยอุตุนิยมวิทยาจริง ๆ

    กฎคร่าว ๆ (เป็น logistic):
      logit(p) = b0
               + b1*อุณหภูมิ     (บวก: ยิ่งร้อนยิ่งเสี่ยง)
               + b2*ความชื้น     (ลบ:  ยิ่งชื้นน้อยยิ่งเสี่ยง)
               + b3*ฝนสะสม7วัน  (ลบ:  ฝนน้อย→แห้ง→เสี่ยง)
               + b4*ความเร็วลม  (บวก: ลมแรง→ไฟลุกลาม)
    """
    # คำนวณฝนสะสม 7 วัน ชั่วคราว (สำหรับสร้าง label)
    rain_7d = df_weather["rain"].rolling(7, min_periods=1).sum().values

    t = df_weather["temperature"].values
    h = df_weather["humidity"].values
    w = df_weather["wind_speed"].values

    # สัมประสิทธิ์ (ปรับให้ signal ชัดเจน เพื่อให้โมเดลเรียนรู้ได้จริง)
    # b0 ตั้งให้ base rate ~1% ในสภาพปกติ; สภาพเสี่ยงสูงขึ้นไป ~40-60%
    b0, b1, b2, b3, b4 = -4.5, 0.18, -0.06, -0.025, 0.10

    logit = b0 + b1 * (t - 28) + b2 * (h - 60) + b3 * (rain_7d - 10) + b4 * (w - 10)
    prob = 1 / (1 + np.exp(-logit))            # sigmoid
    prob = np.clip(prob + fire_base_rate, 0, 0.95)  # ปรับระดับโดยรวม

    # สุ่มตามความน่าจะเป็น
    fire = (rng.random(len(df_weather)) < prob).astype(int)

    # จำนวน hotspot ถ้ามีไฟ → สุ่ม 1-8 จุด
    hotspots = np.where(fire == 1, rng.integers(1, 9, size=len(df_weather)), 0)

    df = df_weather.copy()
    df["fire"] = fire
    df["hotspots"] = hotspots
    df["fire_prob_true"] = np.round(prob, 3)   # เก็บไว้เช็ค (ไม่ใช้ตอน train)
    return df


def generate_dataset(random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """
    สร้างชุดข้อมูลตัวอย่างทั้งหมด

    คืนค่า DataFrame รูปแบบ "grid cell × วัน":
        province, cell_id, lat, lon, date,
        temperature, humidity, rain, wind_speed,
        fire, hotspots
    """
    rng = np.random.default_rng(random_state)

    # สร้างช่วงวันที่ทุกวันตลอด START_YEAR..END_YEAR
    dates = pd.date_range(
        start=f"{config.START_YEAR}-01-01",
        end=f"{config.END_YEAR}-12-31",
        freq="D",
    )

    all_rows = []
    for province, info in config.PROVINCES.items():
        n_cells = info["grid_cells"]
        w, s, e, n = info["bbox"]

        # สุ่มพิกัดศูนย์กลางของแต่ละ cell ภายใน bbox ของจังหวัด
        lats = rng.uniform(s, n, size=n_cells)
        lons = rng.uniform(w, e, size=n_cells)

        for cell_id, (lat, lon) in enumerate(zip(lats, lons), start=1):
            # แต่ละ cell มี base_temp ต่างกันเล็กน้อย (สมจริง)
            base_temp = rng.normal(28, 1.0)
            df_cell = _generate_weather_one_cell(rng, dates, base_temp=base_temp)
            df_cell = _generate_fire_label(df_cell, rng)

            df_cell.insert(0, "province", province)
            df_cell.insert(1, "cell_id", f"{province[:3]}-{cell_id:02d}")
            df_cell.insert(2, "lat", round(lat, 4))
            df_cell.insert(3, "lon", round(lon, 4))
            all_rows.append(df_cell)

    df = pd.concat(all_rows, ignore_index=True)
    # เรียงตามจังหวัด → cell → วันที่
    df = df.sort_values(["province", "cell_id", "date"]).reset_index(drop=True)
    return df


def save_dataset(df: pd.DataFrame, filename: str = "wildfire_sample.csv") -> Path:
    """บันทึกข้อมูลลง data/processed/"""
    out_path = config.PROCESSED_DIR / filename
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ บันทึกข้อมูลที่: {out_path}")
    print(f"   จำนวนแถว: {len(df):,}  |  จำนวนคอลัมน์: {df.shape[1]}")
    print(f"   จำนวนวันที่เกิดไฟ: {df['fire'].sum():,} "
          f"({df['fire'].mean()*100:.1f}%)")
    return out_path


def load_dataset(filename: str = "wildfire_sample.csv") -> pd.DataFrame:
    """โหลดข้อมูลจาก data/processed/"""
    path = config.PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ {path}\n"
            "ให้รัน synthetic_data.py หรือโน้ตบุ๊ก 01_data_generation.ipynb ก่อน"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"✅ โหลดข้อมูลจาก: {path}  ({len(df):,} แถว)")
    return df


# ============================================================
# รันได้โดยตรง:  python synthetic_data.py
# ============================================================
if __name__ == "__main__":
    print("กำลังสร้างข้อมูลตัวอย่าง...")
    df = generate_dataset()
    save_dataset(df)
    print("\nตัวอย่างข้อมูล 5 แถวแรก:")
    print(df.head())
