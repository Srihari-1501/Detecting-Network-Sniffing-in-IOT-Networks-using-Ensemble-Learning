# 🛡️ Detecting Network Sniffing in IoT Wi-Fi Networks using Ensemble Learning

This project presents a machine learning-based solution to detect network sniffing attacks in smart home Wi-Fi environments. As traditional tools often miss these subtle threats, we leveraged ensemble learning models for enhanced accuracy and reliability. The approach includes comprehensive data preprocessing, model training, and a user-friendly interface using Streamlit.

---

## 📁 Project Files

- `DNSI.ipynb` – Jupyter notebook for preprocessing, training, and evaluating ML models.
- `app.py` – Streamlit-based GUI for manual or file-based sniffing detection.
- `saved_models/` – Folder containing trained model files:
  - `rf_model.pkl`  
  - `lgb_model.pkl`  
  - `xgb_model.pkl`  
  - `label_encoder.pkl`

---

## 🧪 Models Used

We tested multiple classification models for their effectiveness:
- ✅ **Decision Tree**
- ✅ **Random Forest**
- ✅ **XGBoost**
- ✅ **LightGBM**

These models showed accuracy above **98%**, with Random Forest performing the best.

---

## ⚙️ Features Used (7-feature subset)

- `CPU_Total`
- `Disk_wrVolume`
- `Disk_wraccessVolume`
- `Mem_Avail`
- `Mem_freeTotal`
- `Traffic_inVolume`
- `Traffic_outVolume`

---

## 🚀 Running the App

### 1. Clone this repository

```bash
git clone https://github.com/<your-username>/iot-sniffing-detection.git
cd iot-sniffing-detection
