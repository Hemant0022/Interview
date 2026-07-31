"""
report_pdf.py
=============
One-click Candidate Report PDF: Resume Analysis, Interview Analysis,
Technical/Communication scores, Roadmap, Final Recommendation.
"""
import io
from typing import Dict

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)


def build_candidate_report(candidate: Dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1c", parent=styles["Heading1"], textColor=colors.HexColor("#1E293B")))
    styles.add(ParagraphStyle(name="H2c", parent=styles["Heading2"], textColor=colors.HexColor("#4F46E5"), spaceBefore=14))

    story = []
    name = candidate.get("name", "Candidate")
    story.append(Paragraph(f"Candidate Report — {name}", styles["H1c"]))
    story.append(Spacer(1, 6))

    resume_score = candidate.get("resume_score", 0.0) or 0.0
    interview_score = candidate.get("interview_score", 0.0) or 0.0
    overall = round(0.5 * resume_score + 0.5 * interview_score, 1) if (resume_score and interview_score) else round(resume_score or interview_score, 1)

    summary_table = Table(
        [["Resume Score", "Interview Score", "Overall Score"],
         [f"{resume_score}%", f"{interview_score}%", f"{overall}%"]],
        colWidths=[1.8 * inch] * 3,
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2F6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 16))

    # ---- Resume analysis ----
    resume_analysis = candidate.get("resume_analysis")
    if resume_analysis:
        story.append(Paragraph("Resume Analysis", styles["H2c"]))
        story.append(Paragraph(f"Hiring-style read: <b>{resume_analysis.get('hiring_recommendation','—')}</b>", styles["Normal"]))
        story.append(Paragraph(f"Job Match Score: {resume_analysis.get('job_match_score','—')}%", styles["Normal"]))
        sw = resume_analysis.get("strengths_weaknesses", {})
        if sw.get("strengths"):
            story.append(Paragraph("Strengths: " + "; ".join(sw["strengths"]), styles["Normal"]))
        if sw.get("weaknesses"):
            story.append(Paragraph("Weaknesses: " + "; ".join(sw["weaknesses"]), styles["Normal"]))
        story.append(Spacer(1, 8))

    # ---- Interview analysis ----
    interview_report = candidate.get("interview_report")
    if interview_report:
        story.append(Paragraph("Interview Analysis", styles["H2c"]))
        story.append(Paragraph(f"Overall Interview Score: {interview_report.get('overall_score','—')}%", styles["Normal"]))
        agg = interview_report.get("aggregate_scores", {})
        if agg:
            rows = [["Dimension", "Score"]] + [[d.replace("_", " ").title(), f"{s}%"] for d, s in agg.items()]
            t = Table(rows, colWidths=[3 * inch, 1.5 * inch])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F8F9FC")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
            ]))
            story.append(Spacer(1, 6))
            story.append(t)
        story.append(Spacer(1, 8))

    # ---- Roadmap ----
    roadmap = candidate.get("roadmap")
    if roadmap:
        story.append(Paragraph("Learning Roadmap", styles["H2c"]))
        for week in roadmap.get("weeks", []):
            story.append(Paragraph(f"<b>Week {week['week']} — {week['title']}</b>", styles["Normal"]))
            for task in week.get("tasks", []):
                story.append(Paragraph(f"• {task['topic']}: {task.get('why','')}", styles["Normal"]))
        story.append(Spacer(1, 8))

    # ---- Final recommendation ----
    story.append(Paragraph("Final Recommendation", styles["H2c"]))
    if overall >= 80:
        rec = "Hire"
    elif overall >= 65:
        rec = "Consider"
    elif overall >= 45:
        rec = "Upskill"
    else:
        rec = "Not a fit"
    story.append(Paragraph(f"<b>{rec}</b> — based on a combined resume + interview score of {overall}%.", styles["Normal"]))

    doc.build(story)
    return buf.getvalue()
