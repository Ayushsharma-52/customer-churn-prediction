import joblib, pandas as pd, streamlit as st

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")
art = joblib.load("models/churn_model.joblib")
model, thr = art["model"], art["threshold"]

st.title("📉 Customer Churn Predictor")
st.caption(f"Model: {art['model_name']} | decision threshold: {thr:.2f} (cost-optimised)")

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Account")
    tenure = st.slider("Tenure (months)", 0, 72, 6)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges", 15.0, 125.0, 80.0)
with c2:
    st.subheader("Services")
    phone = st.selectbox("Phone service", ["Yes", "No"])
    lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    opts = ["No", "Yes", "No internet service"]
    sec = st.selectbox("Online security", opts); bkp = st.selectbox("Online backup", opts)
    prot = st.selectbox("Device protection", opts); tech = st.selectbox("Tech support", opts)
    tv = st.selectbox("Streaming TV", opts); mov = st.selectbox("Streaming movies", opts)
with c3:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = st.selectbox("Partner", ["Yes", "No"]); dep = st.selectbox("Dependents", ["No", "Yes"])

row = pd.DataFrame([{
    "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dep, "tenure": tenure,
    "PhoneService": phone, "MultipleLines": lines, "InternetService": internet, "OnlineSecurity": sec,
    "OnlineBackup": bkp, "DeviceProtection": prot, "TechSupport": tech, "StreamingTV": tv,
    "StreamingMovies": mov, "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": monthly * tenure}])

if st.button("Predict churn risk", type="primary"):
    p = float(model.predict_proba(row)[0, 1])
    st.metric("Churn probability", f"{p:.1%}")
    st.progress(min(p, 1.0))
    if p >= thr:
        st.error("High risk: flag for a retention offer.")
    else:
        st.success("Low risk under the current threshold.")
    if p >= thr:
        tips = []
        if contract == "Month-to-month": tips.append("Offer a discount to move to a 1 or 2-year contract.")
        if tech == "No" and internet != "No": tips.append("Offer free Tech Support for a trial period.")
        if payment == "Electronic check": tips.append("Offer a bill credit for switching to automatic payment.")
        if internet == "Fiber optic": tips.append("Check service quality and consider a price lock.")
        for t in tips: st.write("•", t)
