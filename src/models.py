"""
models.py
=========
สร้าง ฝึก ประเมิน และบันทึกโมเดลประเมินความเสี่ยงไฟป่า

โมเดล 2 แบบ:
1. Logistic Regression  - อธิบายได้ (บอกได้ว่าปัจจัยเพิ่ม/ลดความเสี่ยงอย่างไร)
2. Random Forest        - ทำนายแม่นกว่า + บอกความสำคัญของแต่ละปัจจัย

การแบ่ง train/test:
- แบ่งตามเวลา (temporal split) ไม่ใช่สุ่ม
- train = ปี 2022-2023,  test = ปี 2024
- เหตุผล: กัน data leakage (โมเดล "แอบเห็น" ข้อมูลอนาคตทำให้ผลเฟ้อ)
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report,
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import config


# ============================================================
# 1. แบ่งข้อมูล train/test ตามเวลา
# ============================================================
def temporal_split(df: pd.DataFrame):
    """
    แบ่ง train/test ตามปี (ป้องกัน data leakage)
    - train: ปี <= TRAIN_END_YEAR (2023)
    - test : ปี >= TEST_START_YEAR (2024)
    """
    df = df.copy()
    df["_year"] = df["date"].dt.year

    train_df = df[df["_year"] <= config.TRAIN_END_YEAR].drop(columns="_year")
    test_df  = df[df["_year"] >= config.TEST_START_YEAR].drop(columns="_year")

    feature_cols = config.FEATURE_COLS
    X_train = train_df[feature_cols]
    y_train = train_df[config.TARGET_COL]
    X_test  = test_df[feature_cols]
    y_test  = test_df[config.TARGET_COL]

    print(f"Train: {len(train_df):,} แถว (ปี {config.START_YEAR}-{config.TRAIN_END_YEAR})  "
          f"อัตราไฟ = {y_train.mean()*100:.1f}%")
    print(f"Test : {len(test_df):,} แถว (ปี {config.TEST_START_YEAR}-{config.END_YEAR})  "
          f"อัตราไฟ = {y_test.mean()*100:.1f}%")
    return X_train, X_test, y_train, y_test


# ============================================================
# 2. สร้างโมเดล
# ============================================================
def build_logistic() -> Pipeline:
    """
    Logistic Regression พร้อม standardize
    (ต้อง standardize เพราะแต่ละตัวแปรมีสเกลต่างกัน เช่น อุณหภูมิ vs ฝน)
    ใช้ class_weight='balanced' เพื่อรับมือ class imbalance (ไฟน้อยกว่าไม่ไฟ)
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        )),
    ])


def build_random_forest() -> RandomForestClassifier:
    """
    Random Forest
    - ไม่ต้อง standardize (tree-based ไม่สนสเกล)
    - class_weight='balanced_subsample' รับมือ imbalance แรงๆ (ไฟ ~6% เทียบกับไม่ไฟ 94%)
    - max_depth จำกัดเพื่อกัน overfit (RF มีแนวโน้มทำนาย "ไม่ไฟ" ตลอดเวลา imbalance สูง)
    """
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )


# ============================================================
# 3. ประเมินผลโมเดล
# ============================================================
def evaluate_model(model, X_test, y_test, name: str = "Model") -> dict:
    """
    ประเมินและคืนค่าเมตริกทั้งหมดเป็น dict
    พิมพ์ตารางผลให้ดูด้วย
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "pr_auc":    average_precision_score(y_test, y_prob),  # สำคัญเมื่อ imbalance
    }

    print(f"\n{'='*55}")
    print(f" ผลประเมิน: {name}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {metrics['accuracy']:.3f}")
    print(f"  Precision : {metrics['precision']:.3f}  (ทำนายไฟแล้วเป็นไฟจริงกี่ %)")
    print(f"  Recall    : {metrics['recall']:.3f}  (จับไฟจริงได้กี่ %)")
    print(f"  F1-score  : {metrics['f1']:.3f}  (balance precision & recall)")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.3f}  (overall ≥ 0.5)")
    print(f"  PR-AUC    : {metrics['pr_auc']:.3f}  (สำคัญตอน class imbalance)")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix (แถว=จริง, คอลัมน์=ทำนาย):")
    print(f"                 ทำนายไม่ไฟ   ทำนายไฟ")
    print(f"    จริงไม่ไฟ     {cm[0,0]:>8}   {cm[0,1]:>7}")
    print(f"    จริงเป็นไฟ     {cm[1,0]:>8}   {cm[1,1]:>7}")
    return metrics


def get_feature_importance(model, feature_names) -> pd.DataFrame:
    """
    ดึงค่าความสำคัญของแต่ละตัวแปรจากโมเดล
    รองรับทั้ง Logistic (|coef|) และ RandomForest (feature_importances_)
    """
    # กรณีเป็น Pipeline ให้ดึงขั้นตอนสุดท้าย
    est = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    if hasattr(est, "feature_importances_"):
        importance = est.feature_importances_
        method = "feature_importances_"
    elif hasattr(est, "coef_"):
        # Logistic: ใช้ค่าสัมบูรณ์ของสัมประสิทธิ์
        importance = np.abs(est.coef_[0])
        method = "|coef|"
    else:
        return pd.DataFrame({"feature": feature_names, "importance": 0})

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
        "method": method,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


# ============================================================
# 4. บันทึก / โหลดโมเดล
# ============================================================
def save_model(model, filename: str):
    """บันทึกโมเดลลง models/"""
    path = config.MODELS_DIR / filename
    joblib.dump(model, path)
    print(f"✅ บันทึกโมเดลที่: {path}")
    return path


def load_model(filename: str):
    """โหลดโมเดลจาก models/"""
    path = config.MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล {path}")
    return joblib.load(path)


# ============================================================
# 5. Pipeline หลัก: เทรนทั้ง 2 โมเดลพร้อมกัน
# ============================================================
def train_all(df: pd.DataFrame, save: bool = True) -> dict:
    """
    เทรนทั้ง Logistic และ RandomForest พร้อมกัน
    คืนค่า dict: { "logistic": {"model":..., "metrics":...}, "rf": {...} }
    """
    print("\n" + "🔥"*28)
    print("  เริ่มเทรนโมเดลประเมินความเสี่ยงไฟป่า")
    print("🔥"*28)

    X_train, X_test, y_train, y_test = temporal_split(df)

    results = {}

    # --- 1) Logistic Regression ---
    print("\n[1/2] เทรน Logistic Regression...")
    logit = build_logistic()
    logit.fit(X_train, y_train)
    metrics_logit = evaluate_model(logit, X_test, y_test, "Logistic Regression")
    results["logistic"] = {
        "model": logit,
        "metrics": metrics_logit,
        "importance": get_feature_importance(logit, config.FEATURE_COLS),
    }

    # --- 2) Random Forest ---
    print("\n[2/2] เทรน Random Forest...")
    rf = build_random_forest()
    rf.fit(X_train, y_train)
    metrics_rf = evaluate_model(rf, X_test, y_test, "Random Forest")
    results["rf"] = {
        "model": rf,
        "metrics": metrics_rf,
        "importance": get_feature_importance(rf, config.FEATURE_COLS),
    }

    # --- สรุปเปรียบเทียบ ---
    print("\n" + "="*55)
    print(" สรุปเปรียบเทียบโมเดล")
    print("="*55)
    cmp = pd.DataFrame([results["logistic"]["metrics"],
                        results["rf"]["metrics"]]).set_index("model")
    print(cmp.round(3).to_string())

    # เลือกโมเดลที่ดีที่สุดด้วย ROC-AUC สำหรับแดชบอร์ด
    best_name = "rf" if metrics_rf["roc_auc"] >= metrics_logit["roc_auc"] else "logistic"
    print(f"\n🏆 โมเดลที่ดีที่สุด (ตาม ROC-AUC): {results[best_name]['metrics']['model']}")

    if save:
        save_model(results["logistic"]["model"], "logistic_model.joblib")
        save_model(results["rf"]["model"], "rf_model.joblib")
        # บันทึกตารางความสำคัญไว้ใช้ในแดชบอร์ด
        results[best_name]["importance"].to_csv(
            config.MODELS_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig"
        )
        # บันทึกชื่อโมเดลที่ดีที่สุด
        (config.MODELS_DIR / "best_model.txt").write_text(best_name, encoding="utf-8")
        print(f"✅ บันทึก feature importance + ชื่อโมเดลที่ดีที่สุด ({best_name})")

    return results


# ============================================================
if __name__ == "__main__":
    from synthetic_data import load_dataset
    from features import build_features

    df = load_dataset()
    df = build_features(df)
    results = train_all(df)
