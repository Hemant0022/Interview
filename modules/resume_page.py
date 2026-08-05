import tempfile
import os
import re

import streamlit as st

from backend.resume_engine import analyze, load_document
from backend import candidate_store

def render_structured_section(section_content: str, default_msg: str):
    if not section_content or "not clearly detected" in section_content.lower() or default_msg.strip("_ ") in section_content:
        st.markdown(f"*{default_msg.strip('_ ')}*")
        return
        
    lines = [line.strip() for line in section_content.split("\n") if line.strip()]
    if not lines:
        st.markdown(f"*{default_msg.strip('_ ')}*")
        return
        
    date_regex = re.compile(
        r'\(?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\s*\d{0,2}\s*(?:19\d{2}|20\d{2})\s*[-–—]\s*(?:Present|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\s*\d{0,2}\s*(?:19\d{2}|20\d{2}|\d{2})\)?'
        r'|\(?(?:19\d{2}|20\d{2})\s*[-–—]\s*(?:Present|19\d{2}|20\d{2}|\d{2})\)?'
        r'|\((?:19\d{2}|20\d{2})\)'
        r'|\b(?:19\d{2}|20\d{2})\b',
        re.IGNORECASE
    )
    
    for line in lines:
        # Strip leading bullets or numbers
        cleaned_line = re.sub(r'^[-*•o🔹▪\s]|\d+\.\s*', '', line).strip()
        if not cleaned_line:
            continue
            
        # Try to extract date
        date_match = date_regex.search(cleaned_line)
        date_str = ""
        if date_match:
            date_str = date_match.group(0).strip("() ")
            cleaned_line = date_regex.sub("", cleaned_line).strip()
            
        cleaned_line = cleaned_line.rstrip(",;-—–| ").lstrip(",;-—–| ").strip()
        
        # Split remaining line by common separators
        parts = re.split(r'\s*[|•·]\s+|\s*,\s*|\s+[-–—]\s+', cleaned_line)
        parts = [p.strip() for p in parts if p.strip()]
        
        title = ""
        org = ""
        
        if len(parts) >= 2:
            title = parts[0]
            org = parts[1]
            if len(parts) > 2:
                org += f" ({', '.join(parts[2:])})"
        elif len(parts) == 1:
            title = parts[0]
        else:
            title = cleaned_line
            
        org_html = f'<div style="font-size: 13px; color: #64748B; margin-top: 4px;">{org}</div>' if org else ""
        date_html = f'<div style="background: #EEF2F6; color: #4F46E5; font-size: 11px; font-weight: 600; padding: 4px 10px; border-radius: 20px; white-space: nowrap;">📅 {date_str}</div>' if date_str else ""
        
        card_html = f"""
        <div style="
            background: #FFFFFF; 
            border: 1px solid #E2E8F0; 
            border-radius: 12px; 
            padding: 16px; 
            margin-bottom: 12px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        ">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 8px;">
                <div style="flex: 1; min-width: 200px;">
                    <div style="font-size: 15px; font-weight: 700; color: #1E293B;">{title}</div>
                    {org_html}
                </div>
                {date_html}
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)


def render_match_missing(matched, missing, feedback, matched_empty="No matches found.", missing_empty="Nothing missing."):
    """Render a Matched / Missing block with clear visual separation between
    each heading, and cards that show the actual matched/missing text
    (not just a generic 'matched' / 'not matched' label)."""

    st.markdown('<div class="subsection-title">✅ Matched</div>', unsafe_allow_html=True)
    if matched:
        for item in matched:
            st.markdown(f'<div class="match-card match-good">{item}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="match-card match-neutral">{matched_empty}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="tab-divider">', unsafe_allow_html=True)

    st.markdown('<div class="subsection-title">⚠️ Missing</div>', unsafe_allow_html=True)
    if missing:
        for item in missing:
            st.markdown(f'<div class="match-card match-bad">{item}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="match-card match-good">{missing_empty}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="tab-divider">', unsafe_allow_html=True)

    if feedback:
        st.info(feedback)

# ---------------------------------------------------------------------------


def render():
    """Candidate > Upload Resume & Resume Analysis page."""
    # PAGE CONFIG
    # ---------------------------------------------------------------------------
    st.markdown("""
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1100px;}
    .metric-card {
        background: #F8F9FC; border: 1px solid #E7E9F3; border-radius: 16px;
        padding: 18px 20px; text-align: center;
    }
    .metric-card .value {font-size: 30px; font-weight: 700; color:#181A2A;}
    .metric-card .label {font-size: 13px; color:#6B7280; margin-top:4px;}
    .chip {
        display:inline-block; padding:5px 12px; border-radius:20px;
        font-size:12.5px; font-weight:600; margin:3px;
    }
    .chip-good {background:#E7FBF1; color:#12B76A;}
    .chip-bad {background:#FEECEB; color:#F04438;}
    .section-title {
        font-size:17px; font-weight:700; margin:32px 0 10px;
        padding-top:20px; border-top:1px solid #E2E8F0;
    }
    .small-note {color:#6B7280; font-size:12.5px;}
    .subsection-title {
        font-size:14px; font-weight:700; letter-spacing:0.2px;
        margin:4px 0 10px; color:#1E293B;
    }
    .match-card {
        border-radius:10px; padding:10px 14px; margin-bottom:8px;
        font-size:14px; line-height:1.5; border-left:4px solid transparent;
    }
    .match-good {background:#F0FDF6; color:#0F5132; border-left-color:#12B76A;}
    .match-bad {background:#FEF2F2; color:#7A1F1A; border-left-color:#F04438;}
    .match-neutral {background:#F1F5F9; color:#475569; border-left-color:#94A3B8;}
    hr.tab-divider {
        border:none; border-top:1px solid #E2E8F0; margin:18px 0 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # HEADER — framed as a personal prep tool, not a hiring dashboard
    # ---------------------------------------------------------------------------
    st.title("🧭 Resume Readiness Check")
    st.write(
        "See how your resume stacks up against a job description before your "
        "interview — the same checks an applicant tracking system would run, "
        "so there are no surprises later."
    )

    # ---------------------------------------------------------------------------
    # INPUTS
    # ---------------------------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your resume")
        resume_file = st.file_uploader(
            "Upload your resume (PDF or DOCX)", type=["pdf", "docx"], key="resume"
        )

    with col2:
        st.subheader("Job description")
        jd_input_mode = st.radio(
            "How do you want to add the job description?",
            ["Paste text", "Upload file"],
            horizontal=True,
            label_visibility="collapsed",
        )
        jd_text = ""
        if jd_input_mode == "Paste text":
            jd_text = st.text_area(
                "Paste the job description here",
                height=200,
                placeholder="Paste the full job posting you're applying to…",
            )
        else:
            jd_file = st.file_uploader(
                "Upload the job description (PDF or DOCX)", type=["pdf", "docx"], key="jd"
            )
            if jd_file is not None:
                suffix = os.path.splitext(jd_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(jd_file.read())
                    jd_text = load_document(tmp.name)
                st.text_area("Preview", jd_text, height=150, disabled=True)

    run = st.button("Check my resume", type="primary", use_container_width=False)
    result = st.session_state.get("resume_result_obj")
    analysis = result.section_analysis if result is not None else None

    # ---------------------------------------------------------------------------
    # ANALYSIS
    # ---------------------------------------------------------------------------
    if run:
        if resume_file is None or not jd_text.strip():
            st.warning("Add both your resume and a job description to run the check.")
            st.stop()

        with st.spinner("Reading your resume and comparing it to the job description…"):
            suffix = os.path.splitext(resume_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(resume_file.read())
                resume_text = load_document(tmp.name)

            result = analyze(resume_text, jd_text)
            analysis = result.section_analysis
        st.session_state.resume_result_obj = result
        st.session_state.resume_result = result.to_dict()
        st.success("Done — here's how your resume compares.")

    if run or result is not None:
        # ---- Candidate profile (what the tool read from your resume) ----
        st.markdown('<div class="section-title">What we read from your resume</div>', unsafe_allow_html=True)
    
        name = result.candidate.get("name", "Unknown Candidate")
        initials = "".join([part[0].upper() for part in name.split() if part][:2])
        experience = f"{result.candidate['years_experience']} yrs" if result.candidate.get("years_experience") else "Not found"
        education = result.candidate.get("highest_education", "Not specified")
        email = result.candidate.get("email", "Not found")
        phone = result.candidate.get("phone", "Not found")
    
        profile_html = f"""
        <div style="
            background: #F8F9FC; 
            border: 1px solid #E7E9F3; 
            border-radius: 16px; 
            padding: 24px; 
            margin-bottom: 20px;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
                <div style="display: flex; align-items: center; gap: 16px;">
                    <div style="
                        background: #EEF2F6; 
                        border-radius: 50%; 
                        width: 50px; 
                        height: 50px; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center;
                        font-size: 20px;
                        color: #4F46E5;
                        font-weight: 700;
                    ">
                        {initials}
                    </div>
                    <div>
                        <h3 style="margin: 0; font-size: 20px; font-weight: 700; color: #1E293B;">{name}</h3>
                        <p style="margin: 2px 0 0 0; font-size: 13px; color: #64748B;">Candidate Profile</p>
                    </div>
                </div>
                <div style="display: flex; gap: 24px; flex-wrap: wrap;">
                    <div style="border-left: 3px solid #6366F1; padding-left: 12px;">
                        <div style="font-size: 11px; color: #64748B; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Experience</div>
                        <div style="font-size: 14px; font-weight: 700; color: #0F172A; margin-top: 2px;">{experience}</div>
                    </div>
                    <div style="border-left: 3px solid #10B981; padding-left: 12px;">
                        <div style="font-size: 11px; color: #64748B; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Highest Education</div>
                        <div style="font-size: 14px; font-weight: 700; color: #0F172A; margin-top: 2px;">{education}</div>
                    </div>
                    <div style="border-left: 3px solid #F59E0B; padding-left: 12px;">
                        <div style="font-size: 11px; color: #64748B; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Email</div>
                        <div style="font-size: 14px; font-weight: 700; color: #0F172A; margin-top: 2px;">{email}</div>
                    </div>
                    <div style="border-left: 3px solid #3B82F6; padding-left: 12px;">
                        <div style="font-size: 11px; color: #64748B; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Phone</div>
                        <div style="font-size: 14px; font-weight: 700; color: #0F172A; margin-top: 2px;">{phone}</div>
                    </div>
                </div>
            </div>
        </div>
        """
        st.markdown(profile_html, unsafe_allow_html=True)
        st.caption(
            "If any of this looks wrong or missing, an applicant tracking system "
            "will likely misread it too — worth fixing before you apply."
        )

        # ---- Extracted text, section by section ----
        st.markdown('<div class="section-title">Your resume, section by section</div>', unsafe_allow_html=True)
        st.caption(
            "This is exactly what the parser pulled out of your file. If a section "
            "below looks empty, it likely means that heading wasn't clearly labeled "
            "in your resume — worth fixing, since an ATS will have the same trouble."
        )

    
        tabs = st.tabs([
            "🎓 Education", "🛠️ Technical Skills", "🤝 Soft Skills",
            "💼 Experience", "📁 Projects", "📜 Certifications",
        ])

        with tabs[0]:
            edu = analysis["education"]
            render_match_missing(edu["matched"], edu["missing"], edu["feedback"])

        with tabs[1]:
            tech = analysis["technical_skills"]
            render_match_missing(tech["matched"], tech["missing"], tech["feedback"])

        with tabs[2]:
            soft = analysis["soft_skills"]
            render_match_missing(soft["matched"], soft["missing"], soft["feedback"])

        with tabs[3]:
            exp = analysis["experience"]
            render_match_missing(exp["matched"], exp["missing"], exp["feedback"])

        with tabs[4]:
            proj = analysis["projects"]
            render_match_missing(proj["matched"], proj["missing"], proj["feedback"])

        with tabs[5]:
            cert = analysis["certifications"]
            render_match_missing(cert["matched"], cert["missing"], cert["feedback"])
        # ---- Key scores ----
        st.markdown('<div class="section-title">Your scores</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="value">{result.job_match_score}%</div><div class="label">Job Match Score</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="value">{result.resume_completeness["score"]}%</div><div class="label">Resume Completeness</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="value">{result.resume_structure["score"]}%</div><div class="label">Resume Structure</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="value" style="font-size:18px">{result.interview_readiness["level"]}</div><div class="label">Interview Readiness</div></div>', unsafe_allow_html=True)

        st.info(result.interview_readiness["summary"])
        st.caption(f"Hiring-style read: **{result.hiring_recommendation}**")

        # ---- Score breakdown ----
        st.markdown('<div class="section-title">Job Match Score breakdown</div>', unsafe_allow_html=True)
        st.caption(
            "Skills 35% · Experience 20% · Education 10% · Resume↔JD Semantic Similarity 20% · Project Quality 15%"
        )
        sb1, sb2, sb3, sb4, sb5 = st.columns(5)
        with sb1:
            st.write(f"🛠️ **Skills**: {result.skills_score}%")
            st.progress(min(1.0, result.skills_score / 100))
        with sb2:
            st.write(f"💼 **Experience**: {result.experience_score}%")
            st.progress(min(1.0, result.experience_score / 100))
        with sb3:
            st.write(f"🎓 **Education**: {result.education_score}%")
            st.progress(min(1.0, result.education_score / 100))
        with sb4:
            st.write(f"🧠 **Semantic Match**: {result.semantic_similarity['score']}%")
            st.progress(min(1.0, result.semantic_similarity['score'] / 100))
        with sb5:
            st.write(f"📁 **Project Quality**: {result.project_analysis['score']}%")
            st.progress(min(1.0, result.project_analysis['score'] / 100))
        st.caption(f"Semantic similarity method: {result.semantic_similarity['method']}")

        # ---- Skills comparison (categorized, not duplicated) ----
        st.markdown('<div class="section-title">Skills this job is looking for</div>', unsafe_allow_html=True)
        match_icon = {"exact": "✓", "synonym": "≈", "semantic": "~"}
        sc1, sc2 = st.columns(2)
        with sc1:
            st.write("**Skills your resume already shows for this role**")
            chips = "".join(
                f'<span class="chip chip-good">{match_icon.get(m["match_type"], "✓")} {m["skill"]}</span>'
                for m in result.skill_match["matched"]
            )
            st.markdown(chips or "_None detected yet_", unsafe_allow_html=True)
            st.caption("✓ exact match · ≈ synonym/abbreviation match · ~ semantic match")
        with sc2:
            st.write("**Skills to add or highlight**")
            st.markdown(
                "".join(f'<span class="chip chip-bad">{s}</span>' for s in result.skill_match["missing"])
                or "_Nothing missing — nice work_",
                unsafe_allow_html=True,
            )

        with st.expander("Your skills, categorized"):
            for category, skills in result.resume_skills_by_category.items():
                st.write(f"**{category}**: " + ", ".join(skills))

        # ---- Resume completeness & structure ----
        st.markdown('<div class="section-title">Resume completeness & structure</div>', unsafe_allow_html=True)
        rc1, rc2 = st.columns(2)
        with rc1:
            st.write(f"**Completeness — {result.resume_completeness['score']}%**")
            for f in result.resume_completeness["feedback"]:
                st.write(f"- {f}")
        with rc2:
            st.write(f"**Structure — {result.resume_structure['score']}%**")
            for f in result.resume_structure["feedback"]:
                st.write(f"- {f}")

        # ---- Project analysis ----
        st.markdown('<div class="section-title">Project analysis</div>', unsafe_allow_html=True)
        pa = result.project_analysis
        st.write(
            f"Estimated projects: **{pa['project_count_estimate']}** · "
            f"Quantified impact: **{'Yes' if pa['has_quantified_impact'] else 'No'}** · "
            f"Relevant tech mentioned: **{', '.join(pa['relevant_tech_mentioned']) or 'None'}**"
        )
        for f in pa["feedback"]:
            st.write(f"- {f}")

        # ---- Strengths & weaknesses ----
        st.markdown('<div class="section-title">Strengths & weaknesses</div>', unsafe_allow_html=True)
        sw1, sw2 = st.columns(2)
        with sw1:
            st.write("**Strengths**")
            for s in result.strengths_weaknesses["strengths"]:
                st.success(s)
        with sw2:
            st.write("**Weaknesses**")
            for w in result.strengths_weaknesses["weaknesses"]:
                st.error(w)

        # ---- Section-wise recommendations ----
        st.markdown('<div class="section-title">Section-wise recommendations</div>', unsafe_allow_html=True)
        for rec in result.recommendations:
            flag = "🔴" if rec["needs_attention"] else "🟢"
            st.write(f"{flag} **{rec['section']}** — {rec['recommendation']}")

        # ---- Why this score (explainability) ----
        st.markdown('<div class="section-title">Why you got this score</div>', unsafe_allow_html=True)
        for key, item in result.explanation.items():
            label = key.replace("_", " ").capitalize()
            st.write(f"**{label}** — {item['value']}")

        # ---- Interview prep talking points ----
        st.markdown('<div class="section-title">Before you walk into the interview</div>', unsafe_allow_html=True)
        for s in result.interview_readiness["talking_points"]:
            st.write(f"- {s}")

        with st.expander("Full raw output (JSON) — structured for downstream use"):
            st.json(result.to_dict())

        st.divider()
        if st.button("Continue to AI Interview →", type="primary"):
            st.session_state.nav_target = "AI Mock Interview"
            st.rerun()

    else:
        st.markdown('<p class="small-note">Add a resume and a job description above, then select "Check my resume".</p>', unsafe_allow_html=True)