# 🚀 คู่มือ Deploy แดชบอร์ดขึ้น Streamlit Community Cloud (ฟรี 100%)

หลังทำตามคู่มือนี้จบ คุณจะได้ลิงก์เว็บแบบนี้:
```
https://USERNAME-wildfire-dashboard.streamlit.app
```
ที่ครู/เพื่อนเปิดดูได้จากทุกที่

---

## ✅ สิ่งที่ต้องเตรียม

- [ ] บัญชี **GitHub** (สมัครฟรีที่ https://github.com/signup)
- [ ] บัญชี **Streamlit Cloud** (ใช้บัญชี GitHub ล็อกอินได้เลย)
- [ ] โฟลเดอร์ `streamlit_cloud/` (เตรียมไว้ให้แล้ว)

⏱ เวลาโดยประมาณ: **20–30 นาที** (รอบแรก)

---

## 📋 ขั้นตอนที่ 1: สมัคร GitHub + อัปโหลดโค้ด

### 1.1 สมัคร GitHub
1. ไปที่ https://github.com/signup
2. สมัครด้วย email → ตั้ง username + password

### 1.2 สร้าง Repository ใหม่
1. คลิกปุ่ม **+** มุมขวาบน → **New repository**
2. ตั้งค่า:
   - **Repository name:** `wildfire-dashboard`
   - **Description:** (ไม่ใส่ก็ได้)
   - **Visibility:** เลือก **Public** ⚠️ (Streamlit Cloud ฟรีแค่ public)
   - ✅ เลือก **Add a README file**
   - **.gitignore:** None
3. คลิก **Create repository**

### 1.3 อัปโหลดไฟล์ (วิธีเว็บ — ง่ายสุด ไม่ต้องลง git)

1. ในหน้า repo ใหม่ คลิก **Add file** → **Upload files**
2. **ลากไฟล์ต่อไปนี้** จากโฟลเดอร์ `streamlit_cloud/` ไปอัปโหลด:
   ```
   app.py
   requirements.txt
   packages.txt
   README.md          (คลิก "Replace" ถ้าถาม)
   .gitignore
   ```
3. คลิกปุ่ม **Commit changes** (เขียว ๆ) → ใส่ข้อความ "Initial dashboard"

### 1.4 สร้างโฟลเดอร์ย่อยและอัปโหลด

⚠️ GitHub เว็บไม่สามารถสร้างโฟลเดอร์ว่างได้ ต้องทำทีละโฟลเดอร์:

**โฟลเดอร์ `src/`:**
1. คลิก **Add file** → **Create new file**
2. ในช่อง **Name your file** พิมพ์ `src/__init__.py` (ใช้ `/` สร้างโฟลเดอร์)
3. คลิก **Commit new file**

4. ทำซ้ำ: **Add file** → **Upload files** → ลากไฟล์ทั้งหมดใน `streamlit_cloud/src/`
   (`config.py`, `features.py`, `models.py`, `synthetic_data.py`, `visualize.py`, `gee_extraction.py`)

**โฟลเดอร์ `data/processed/`:**
1. **Add file** → **Create new file** → พิมพ์ `data/processed/.gitkeep` → Commit
2. **Add file** → **Upload files** → ลาก `wildfire_sample.csv`

**โฟลเดอร์ `models/`:**
1. **Add file** → **Create new file** → พิมพ์ `models/.gitkeep` → Commit
2. **Add file** → **Upload files** → ลากไฟล์ทั้งหมดใน `models/`
   (`best_model.txt`, `feature_importance.csv`, `logistic_model.joblib`, `rf_model.joblib`)

### ✅ เช็คโครงสร้างสุดท้าย
กดที่ชื่อ repo ด้านบน ควรเห็นโครงสร้างนี้:
```
wildfire-dashboard/
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── features.py
│   ├── models.py
│   ├── synthetic_data.py
│   ├── visualize.py
│   └── gee_extraction.py
├── data/
│   └── processed/
│       └── wildfire_sample.csv
└── models/
    ├── best_model.txt
    ├── feature_importance.csv
    ├── logistic_model.joblib
    └── rf_model.joblib
```

---

## 📋 ขั้นตอนที่ 2: Deploy บน Streamlit Cloud

### 2.1 เข้า Streamlit Cloud
1. ไปที่ https://share.streamlit.io
2. คลิก **Sign in** → ใช้บัญชี **GitHub** ล็อกอิน
3. คลิก **Authorize streamlit** (อนุญาตให้อ่าน repo)

### 2.2 สร้าง App ใหม่
1. คลิก **New app** (ปุ่มสีฟ้า)
2. ตั้งค่า:
   - **Repository:** เลือก `USERNAME/wildfire-dashboard`
   - **Branch:** `main`
   - **Main file path:** `app.py` ✅
   - **App URL:** จะตั้งชื่อเองอัตโนมัติ
3. คลิก **Deploy!** 🔴

### 2.3 รอ Build (5–10 นาที)
- หน้าจอจะแสดง log การติดตั้ง library
- ถ้าเห็น "Your app is up!" และแดชบอร์ดขึ้น = **สำเร็จ!** 🎉

---

## 📋 ขั้นตอนที่ 3: เช็คว่าใช้ได้

เปิดแดชบอร์ดแล้วทดสอบ:
- [ ] ภาษาไทยแสดงถูกต้อง (ไม่เป็น □□□)
- [ ] คลิกหน้า **🎮 ทดลองปรับค่า** → ปรับ slider → ดู % ความเสี่ยงเปลี่ยน
- [ ] คลิกหน้า **📊 ภาพรวม** → เห็นแผนที่จุดสีเขียว/เหลือง/ส้ม/แดง
- [ ] คลิกหน้า **🚨 พื้นที่เฝ้าระวัง** → เห็นตารางเรียงตามความเสี่ยง

ลิงก์เว็บของคุณ:
```
https://USERNAME-wildfire-dashboard.streamlit.app
```

---

## 🔧 แก้ปัญหาที่พบบ่อย

| อาการ | วิธีแก้ |
|-------|--------|
| ภาษาไทยเป็น □□□ | ตรวจว่ามีไฟล์ `packages.txt` ที่ root repo |
| ImportError | ตรวจ `requirements.txt` ว่าครบทุกบรรทัด |
| FileNotFoundError | ตรวจว่าอัป `data/processed/wildfire_sample.csv` และ `models/*.joblib` ครบ |
| ModuleNotFoundError: src | ตรวจว่าโฟลเดอร์ `src/` มี `__init__.py` |
| Build failed | ดูแท็บ **Manage app** → **Logs** แล้วหาบรรทัด Error |
| App ขึ้น "Manage app" แต่จอดำ | คลิก **Manage app → Reboot** หรือรอ 1-2 นาที |

---

## 🔄 อัปเดตเว็บภายหลัง

ถ้าแก้โค้ดในเครื่องแล้วอยาก deploy ใหม่:
1. แก้ไฟล์ใน repo GitHub (คลิกไฟล์ → ไอคอนดินสอ → แก้ → Commit)
2. Streamlit Cloud จะ **redeploy อัตโนมัติ** ภายใน 1-2 นาที
3. หรือไปที่ Streamlit Cloud → **Manage app** → ปุ่ม **Rerun/Reboot**

---

## 🎁 ทริคเพิ่มเติม

- **เปลี่ยนชื่อเว็บ:** แก้บรรทัด `st.set_page_config(page_title=...)` ใน app.py
- **รับการแจ้งเตือนถ้า app down:** Streamlit Cloud → Settings → Email alerts
- **ดูสถิติผู้ใช้:** แท็บ Dashboard ใน Streamlit Cloud
- **App หลับถ้าไม่มีใครใช้ 7 วัน:** ฟรี tier จะ sleep; กด **Reboot** คืนชีพได้

---

## 📞 ลิงก์ที่เป็นประโยชน์

- Streamlit Cloud: https://share.streamlit.io
- เอกสาร: https://docs.streamlit.io/streamlit-community-cloud
- ฟอนต์ไทยใน Streamlit: https://discuss.streamlit.io/t/how-to-use-thai-font/25234

---

🎉 **เสร็จแล้ว! แชร์ลิงก์ให้ครู/เพื่อนดูได้เลย**
