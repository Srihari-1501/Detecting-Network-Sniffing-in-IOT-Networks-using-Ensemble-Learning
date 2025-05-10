import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def inputValues(model, label_encoder):
    st.subheader("🔧 Manual Input Mode")

    cputotal = int(st.number_input("CPU Total"))
    diskwr = st.number_input("Disk Write Volume",format="%.5f")
    diskwra = st.number_input("Disk Write Access Volume",format="%.5f")
    memavail = st.number_input("Available Memory",format="%.0f")

    memfree = st.number_input("Free Memory",format="%.0f")
    trafficin = st.number_input("Traffic In Volume",format="%.5f")
    trafficout = st.number_input("Traffic Out Volume",format="%.5f")
    actual_class = st.text_input("Actual Class")
    predict=st.button('Predict')
    if predict:
        feature_names = [
            'CPU_Total', 'Disk_wrVolume', 'Disk_wraccessVolume',
            'Mem_Avail', 'Mem_freeTotal', 'Traffic_inVolume', 'Traffic_outVolume'
        ]

        manual_input = pd.DataFrame(
            [[cputotal, diskwr, diskwra, memavail, memfree, trafficin, trafficout]],
            columns=feature_names
        )

        pred_encoded = model.predict(manual_input)
        pred_decoded = label_encoder.inverse_transform(pred_encoded)
        
        if pred_decoded:
            st.subheader("Actual Class")
            st.write(actual_class)

            st.subheader("Predicted Class")
            st.write(pred_decoded[0])


def csvValues(model, label_encoder):
    st.subheader(" File Upload Mode")
    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)

        # Drop irrelevant columns
        drop_cols = [
            'CPU_P1', 'CPU_P2', 'CPU_P3', 'CPU_P4',
            'Load_1min', 'Load_5min', 'Load_15min',
            'Ping_Time(ms)', 'Ping_Min(ms)', 'Ping_Max(ms)', 'Ping_PacketLoss'
        ]
        df = df.drop(columns=drop_cols, errors='ignore')  # Use errors='ignore' for robustness

        if 'Class' not in df.columns:
            st.error(" 'Class' column is required for evaluation.")
            return

        features_only = df.drop(columns=['Class'])
        

        if st.button("Predict"):
            pred_encoded = model.predict(features_only)
            pred_decoded = label_encoder.inverse_transform(pred_encoded)

            df['Predicted_Class'] = pred_decoded

            accuracy = accuracy_score(df['Class'], df['Predicted_Class'])

            df.to_csv("predictions.csv", index=False)

            st.success(f"Predictions completed and saved to `predictions.csv`")
            st.write(" **Prediction Accuracy:**", f"{accuracy * 100:.2f}%")
            st.subheader(" Sample Predictions")
            st.dataframe(df[['Class', 'Predicted_Class']].head(10))
            
def run_app():
    st.title("Enhancing Sniffing Detection using Ensemble Learning")

    inputValue = st.checkbox("Use manual input instead of file upload")

    # Load model and encoder
    model = joblib.load('saved_models/lgb_model.pkl')  # or rf_model.pkl
    label_encoder = joblib.load('saved_models/label_encoder.pkl')

    if inputValue:
        inputValues(model, label_encoder)
    else:
        csvValues(model, label_encoder)

if __name__ == "__main__":
    run_app()
