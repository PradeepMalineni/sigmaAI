import streamlit as st
from app.data_loader import load_incident_data
from app.analyzer import IncidentAnalyzer
from app.model_fallback import analyze_with_fallback

st.title("🔍 GenAI Incident Analyzer")
user_input = st.text_area("Describe your issue or paste incident number")

df = load_incident_data()
analyzer = IncidentAnalyzer(df)

if st.button("Analyze"):
    match = df[df["incident_id"] == user_input]
    query_text = match.iloc[0]["combined_text"] if not match.empty else user_input

    similar = analyzer.retrieve_similar(query_text)
    st.subheader("🔍 Similar Incidents")
    st.dataframe(similar[["incident_id", "ci_id", "description", "resolution", "cause"]])

    st.subheader("🧠 Suggested RCA & Resolution")
    with st.spinner("Generating RCA & Resolution..."):
        result = analyze_with_fallback(query_text, similar)

    st.text_area("Output", result.strip(), height=300)