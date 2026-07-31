"""
recruiter_dashboard.py
=======================
Recruiter Dashboard: candidate ranking table + Plotly analytics + one-click
PDF report, reading from backend.candidate_store (populated by the Resume
and Interview pages as candidates go through the flow).
"""
import streamlit as st
import pandas as pd
import plotly.express as px

from backend import candidate_store, report_pdf


def render():
    st.title("👨‍💼 Recruiter Dashboard")

    candidates = candidate_store.list_candidates()
    if not candidates:
        st.info("No candidates yet. Candidates appear here once they complete Resume Analysis and/or the AI Interview.")
        return

    rows = []
    for c in candidates:
        overall = candidate_store.overall_score(c)
        rows.append({
            "id": c["id"],
            "Candidate": c.get("name") or "Unknown",
            "Resume": c.get("resume_score", 0.0) or 0.0,
            "Interview": c.get("interview_score", 0.0) or 0.0,
            "Overall": overall,
            "Recommendation": candidate_store.recommendation_for(overall),
        })
    df = pd.DataFrame(rows)

    # ---------------- Candidate Ranking ----------------
    st.subheader("Candidate Ranking")
    ranked = df.sort_values("Overall", ascending=False).reset_index(drop=True)

    def _style_row(row):
        color = {"Hire": "background-color: #E7FBF1", "Consider": "background-color: #FFF7E6",
                 "Upskill": "background-color: #FFF1EB", "Not a fit": "background-color: #FEECEB"}
        return [color.get(row["Recommendation"], "")] * len(row)

    st.dataframe(
        ranked.drop(columns=["id"]).style.apply(_style_row, axis=1),
        use_container_width=True, hide_index=True,
    )

    # ---------------- Candidate detail + report ----------------
    st.subheader("Candidate Report")
    selected_name = st.selectbox("Select a candidate", ranked["Candidate"].tolist())
    selected_id = ranked.loc[ranked["Candidate"] == selected_name, "id"].iloc[0]
    record = candidate_store.get_candidate(selected_id)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Resume", f"{record.get('resume_score', 0.0) or 0.0}%")
    c2.metric("Interview", f"{record.get('interview_score', 0.0) or 0.0}%")
    agg = (record.get("interview_report") or {}).get("aggregate_scores", {})
    c3.metric("Technical", f"{agg.get('technical_accuracy', 0.0)}%")
    c4.metric("Communication", f"{agg.get('communication', 0.0)}%")

    if st.button("📄 Generate PDF Report", type="primary"):
        pdf_bytes = report_pdf.build_candidate_report(record)
        st.download_button(
            "⬇️ Download Candidate Report",
            data=pdf_bytes,
            file_name=f"{selected_name.replace(' ', '_')}_report.pdf",
            mime="application/pdf",
        )

    # ---------------- Analytics ----------------
    st.subheader("Analytics")
    a1, a2 = st.columns(2)
    with a1:
        fig = px.bar(ranked, x="Candidate", y=["Resume", "Interview"], barmode="group",
                     title="Resume vs Interview Score", color_discrete_sequence=["#4F46E5", "#12B76A"])
        st.plotly_chart(fig, use_container_width=True)
    with a2:
        fig2 = px.pie(ranked, names="Recommendation", title="Recommendation Distribution",
                       color="Recommendation",
                       color_discrete_map={"Hire": "#12B76A", "Consider": "#F59E0B", "Upskill": "#F97316", "Not a fit": "#F04438"})
        st.plotly_chart(fig2, use_container_width=True)

    if any(r.get("interview_report") for r in candidates):
        dims = ["technical_accuracy", "communication", "confidence", "grammar", "completeness", "relevance"]
        dim_rows = []
        for c in candidates:
            agg = (c.get("interview_report") or {}).get("aggregate_scores", {})
            if agg:
                for d in dims:
                    dim_rows.append({"Candidate": c.get("name", "Unknown"), "Dimension": d.replace("_", " ").title(), "Score": agg.get(d, 0)})
        if dim_rows:
            fig3 = px.bar(pd.DataFrame(dim_rows), x="Dimension", y="Score", color="Candidate", barmode="group",
                          title="Interview Skill Distribution")
            st.plotly_chart(fig3, use_container_width=True)
