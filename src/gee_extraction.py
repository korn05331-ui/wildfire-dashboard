"""
gee_extraction.py
=================
แม่แบบ (stub/template) สำหรับดึงข้อมูลจริงจาก Google Earth Engine

★ ตอนนี้เป็นโค้ดตัวอย่าง — ต้อง "สมัคร GEE + authenticate" ก่อนถึงจะรันได้
★ เมื่อนำไปใช้จริง output จะมีรูปแบบเดียวกับ synthetic_data.py เป๊ะ ๆ
  เพื่อให้ทุกส่วนอื่นในโปรเจกต์ (features, models, dashboard) ใช้ได้ทันที

วิธีใช้:
1. สมัคร Google Earth Engine ที่ https://code.earthengine.google.com/ (ฟรี)
2. รัน ee.Authenticate() ครั้งแรก (จะได้ลิงก์มาให้คลิก)
3. เลือก dataset 2 ตัวที่จะใช้ (ดูในตัวแปร ERA5_DATASET และ FIRMS_DATASET)
4. รัน extract_real_data() ในแถบ ๆ ที่ต้องการ
"""

# ============================================================
# ขั้นที่ 0: ติดตั้ง + authenticate
# ============================================================
def setup_gee():
    """
    ติดตั้ง + ยืนยันตัวตนกับ Google Earth Engine
    """
    # ใน Colab ใช้คำสั่งนี้ติดตั้ง:
    # !pip install earthengine-api geemap
    import ee
    try:
        # พยายามใช้ token ที่เคย authenticate แล้ว
        ee.Initialize()
        print("✅ GEE พร้อมใช้งาน")
    except Exception:
        # ถ้ายังไม่ได้ authenticate ให้ทำ (เปิดลิงก์แล้ววาง token)
        ee.Authenticate()
        ee.Initialize()
        print("✅ authenticate สำเร็จ")


# ============================================================
# ขั้นที่ 1: กำหนด dataset ที่จะใช้
# ============================================================
# ERA5-Land: ข้อมูลสภาพอากาศย้อนหลัง ความละเอียด ~10 กม.
#  - temperature_2m : อุณหภูมิ (Kelvin → ลบ 273.15 เป็น °C)
#  - u_component / v_component : ลมแนวนอน/แนวตั้ง → คำนวณความเร็วลม
#  - total_precipitation : ฝน (m → คูณ 1000 เป็นมม.)
#  - ความชื้น: ใช้ dewpoint มาคำนวณ relative humidity
ERA5_DATASET = "ECMWF/ERA5_LAND/MONTHLY_AGGR"  # เปลี่ยนเป็น DAILY_AGGR สำหรับรายวัน

# FIRMS: จุดความร้อนจากดาวเทียม (MODIS + VIIRS)
FIRMS_DATASET = "FIRMS"

# ชื่อ band ใน ERA5 (และวิธีแปลงหน่วย)
ERA5_BANDS = {
    "temperature_2m":   ("temperature", lambda x: x - 273.15),     # K → °C
    "total_precipitation": ("rain", lambda x: x * 1000),           # m → มม.
    "u_component_of_wind_10m": "_u",                               # ใช้คำนวณลม
    "v_component_of_wind_10m": "_v",
    "dewpoint_temperature_2m": "_dew",                             # ใช้คำนวณความชื้น
}


def _wind_speed(u_img, v_img):
    """
    คำนวณความเร็วลมจาก u, v component
    speed = sqrt(u² + v²)  (หน่วย m/s → คูณ 3.6 เป็น กม./ชม.)
    """
    return u_img.multiply(u_img).add(v_img.multiply(v_img)).sqrt().multiply(3.6)


def _relative_humidity(temp_c_img, dew_c_img):
    """
    คำนวณ relative humidity จากอุณหภูมิและจุดน้ำค้าง (Magnus formula)
    RH = 100 * exp((17.625*Td)/(243.04+Td) - (17.625*T)/(243.04+T))
    """
    import ee
    a, b = 17.625, 243.04
    term1 = dew_c_img.multiply(a).divide(dew_c_img.add(b))
    term2 = temp_c_img.multiply(a).divide(temp_c_img.add(b))
    return term1.subtract(term2).exp().multiply(100)


# ============================================================
# ขั้นที่ 2: ดึงข้อมูลเป็นรายวัน ในพื้นที่ที่กำหนด
# ============================================================
def build_daily_image(date_str: str):
    """
    สร้าง image รายวันของสภาพอากาศ (ประกอบหลาย band)
    date_str: "YYYY-MM-DD"
    """
    import ee
    start = ee.Date(date_str)
    end = start.advance(1, "day")

    era5 = (ee.ImageCollection(ERA5_DATASET)
            .filterDate(start, end)
            .first())

    # แปลงหน่วย
    temp_c = era5.select("temperature_2m").subtract(273.15)
    rain_mm = era5.select("total_precipitation").multiply(1000)
    u = era5.select("u_component_of_wind_10m")
    v = era5.select("v_component_of_wind_10m")
    wind = _wind_speed(u, v)
    dew_c = era5.select("dewpoint_temperature_2m").subtract(273.15)
    rh = _relative_humidity(temp_c, dew_c)

    # รวมเป็น image เดียว พร้อมตั้งชื่อ band
    img = (temp_c.rename("temperature")
           .addBands(rain_mm.rename("rain"))
           .addBands(wind.rename("wind_speed"))
           .addBands(rh.rename("humidity")))
    return img


def extract_hotspots(region, start_date: str, end_date: str):
    """
    ดึงจำนวน hotspot ต่อวันในพื้นที่ region
    """
    import ee
    firms = (ee.ImageCollection(FIRMS_DATASET)
             .filterDate(start_date, end_date)
             .filterBounds(region))
    return firms


# ============================================================
# ขั้นที่ 3: export เป็นตาราง CSV (รูปแบบเดียวกับ synthetic)
# ============================================================
def extract_real_data(province_bbox, start_date, end_date, out_csv):
    """
    ★ เมื่อนำไปใช้จริง ★
    ดึงข้อมูลจาก GEE และบันทึกเป็น CSV ที่มี format เดียวกับ synthetic_data.py
    - province_bbox: [W, S, E, N]
    - start_date / end_date: "YYYY-MM-DD"
    - out_csv: path สำหรับบันทึกไฟล์

    หลังจากนี้ใช้ features.build_features() ตามปกติได้เลย
    """
    import ee
    setup_gee()

    region = ee.Geometry.Rectangle(province_bbox, "EPSG:4326", False)

    # วนทีละวันและ sample ค่าอากาศตาม grid cell
    # (ในทางปฏิบัติควรใช้ ee.batch.Export.table.toDrive เพื่อได้ไฟล์ใหญ่)
    print(f"กำลังดึงข้อมูลจาก {start_date} ถึง {end_date}")
    print(f"พื้นที่ bbox: {province_bbox}")
    print("⚠ ใช้ sample() สำหรับข้อมูลน้อย; ข้อมูลมากใช้ Export.table.toDrive")
    print(f"จะบันทึกที่: {out_csv}")
    print()
    print("📌 TODO: เติมโค้ด sampling/grid generation ตามความต้องการ")
    print("📌 ผลลัพธ์ต้องมีคอลัมน์เหล่านี้ (เหมือน synthetic_data):")
    print("     province, cell_id, lat, lon, date,")
    print("     temperature, humidity, rain, wind_speed,")
    print("     fire (0/1), hotspots (int)")


# ============================================================
# เกริ่นเพิ่มเติมสำหรับนักเรียน
# ============================================================
"""
ความแตกต่างสำคัญระหว่างข้อมูลจริง vs synthetic:
------------------------------------------------------------------
| ด้าน          | synthetic_data.py   | gee_extraction.py (จริง)    |
|--------------|---------------------|------------------------------|
| แหล่ง         | สุ่มจากสูตร         | ERA5-Land + FIRMS (จาก GEE)  |
| ฤดูกาล       | สร้างเองด้วย sin/cos | มีอยู่แล้วในข้อมูล            |
| fire label   | สุ่มตาม prob        | ใช้ hotspot ≥ 1 = fire       |
| grid cells   | สุ่มพิกัด           | สร้าง grid จริงจาก bbox       |
| เวลาดึง      | < 1 วินาที         | หลายนาที (อาจใช้ Export)     |

เมื่อใช้ข้อมูลจริงแล้ว ส่วน features.py, models.py, dashboard/app.py
ไม่ต้องแก้ เพราะ output format เหมือนกัน!
"""
