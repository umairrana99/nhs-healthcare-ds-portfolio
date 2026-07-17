"""Streamlit dashboard for hospital readmission risk.

Run with:  streamlit run src/readmission/dashboard.py
(requires a saved model — see ``python -m readmission.train``).
"""

from __future__ import annotations

import streamlit as st

from readmission import service
from readmission.api.schemas import PatientRecord
from readmission.config import get_settings
from readmission.persist import load_model

_AGE_BANDS = ["[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"]


def main() -> None:
    st.set_page_config(page_title="Readmission Risk", page_icon="🏥")
    st.title("🏥 30-Day Readmission Risk")
    st.caption("Decision support only; not a substitute for clinical judgement.")

    with st.form("patient"):
        left, right = st.columns(2)
        with left:
            time_in_hospital = int(st.number_input("Time in hospital (days)", 1, 14, 5))
            num_medications = int(st.number_input("Number of medications", 1, 80, 16))
            number_inpatient = int(st.number_input("Prior inpatient visits", 0, 20, 1))
            number_emergency = int(st.number_input("Prior emergency visits", 0, 20, 0))
            number_diagnoses = int(st.number_input("Number of diagnoses", 1, 16, 9))
        with right:
            age = st.selectbox("Age band", _AGE_BANDS, index=3)
            gender = st.selectbox("Gender", ["Female", "Male"])
            insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
            a1c = st.selectbox("A1C result", ["None", "Norm", ">7", ">8"])
            diag_1 = st.text_input("Primary diagnosis (ICD-9)", "410")
        submitted = st.form_submit_button("Predict")

    if submitted:
        record = PatientRecord(
            time_in_hospital=time_in_hospital,
            num_lab_procedures=40,
            num_procedures=1,
            num_medications=num_medications,
            number_outpatient=0,
            number_emergency=number_emergency,
            number_inpatient=number_inpatient,
            number_diagnoses=number_diagnoses,
            gender=gender,
            admission_type_id=1,
            discharge_disposition_id=1,
            admission_source_id=7,
            diag_1=diag_1,
            age=age,
            insulin=insulin,
            A1Cresult=a1c,
        )
        model = load_model(get_settings().model_path)
        result = service.score(model, record)
        st.metric("Readmission probability", f"{result['probability']:.1%}", result["risk_band"])
        st.subheader("Top contributing factors")
        for factor in result["top_factors"]:
            st.write(
                f"- **{factor['feature']}** — {factor['direction']} risk "
                f"(SHAP {factor['shap']:+.3f})"
            )


if __name__ == "__main__":
    main()
