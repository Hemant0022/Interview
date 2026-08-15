import tempfile
import os
import re
import math
import streamlit as st

from backend.resume_engine import analyze, load_document, calculate_job_match_breakdown
from backend import candidate_store


def _dedent_html(html: str) -> str:
    """Streamlit's markdown renderer follows normal Markdown rules: any
    line indented 4+ spaces is treated as an INDENTED CODE BLOCK and gets
    shown as literal escaped text (with a copy button) instead of being
    parsed as HTML. Multi-line f-string HTML tends to pick up exactly that
    much indentation once it's nested inside another indented f-string --
    which is what happened when several sections were composed into one
    big profile card. Stripping each line's leading whitespace sidesteps
    the rule entirely without changing what actually renders (HTML
    collapses whitespace between tags anyway)."""
    return "\n".join(line.lstrip() for line in html.split("\n"))

_DATE_RE = re.compile(
    r'\(?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\s*\d{0,2}\s*(?:19\d{2}|20\d{2})\s*[-–—]\s*(?:Present|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\s*\d{0,2}\s*(?:19\d{2}|20\d{2}|\d{2})\)?'
    r'|\(?(?:19\d{2}|20\d{2})\s*[-–—]\s*(?:Present|19\d{2}|20\d{2}|\d{2})\)?'
    r'|\((?:19\d{2}|20\d{2})\)'
    r'|\b(?:19\d{2}|20\d{2})\b',
    re.IGNORECASE
)
_URL_RE = re.compile(r'(https?://\S+|www\.\S+)', re.IGNORECASE)
_HTML_TAG_RE = re.compile(r'<[^>]+>')
_LINK_LABEL_RE = re.compile(
    r'^(link|url|github|demo|live\s*demo|live\s*link|project\s*link|repo|repository)\s*:?\s*$',
    re.IGNORECASE,
)
_BULLET_PREFIX_RE = re.compile(r'^[-*•o🔹▪●○]\s*|^\d+[\.\)]\s*')


def _clean_resume_line(line: str, strip_links: bool) -> str:
    """Defensively strips any stray HTML tags (so broken markup in the
    extracted resume text can never leak into the page as literal text)
    and, when requested, drops bare URLs and 'Link:'-style label lines."""
    line = _HTML_TAG_RE.sub("", line)
    line = re.sub(r'\s+', ' ', line).strip()
    if not line:
        return ""
    if strip_links:
        if _LINK_LABEL_RE.match(line.strip(" -:")):
            return ""
        line = _URL_RE.sub("", line).strip(" -–—:|")
        if not line or _LINK_LABEL_RE.match(line.strip(" -:")):
            return ""
    return line


def _highlight_dates(text: str, accent_color: str, accent_bg: str) -> str:
    """Wraps any date/date-range found in a heading line with a small
    inline badge, without moving it out of line or splitting the entry."""
    def _wrap(m):
        d = m.group(0).strip("() ")
        return (
            f'<span style="background:{accent_bg}; color:{accent_color}; font-size:11px; '
            f'font-weight:600; padding:2px 9px; border-radius:20px; margin-left:8px; '
            f'white-space:nowrap;">📅 {d}</span>'
        )
    return _DATE_RE.sub(_wrap, text)


def build_structured_section_html(section_content: str, default_msg: str, accent_color: str = "#4F46E5",
                                   accent_bg: str = "#EEF2F6", strip_links: bool = True) -> str:
    """Builds (but does not render) the HTML for a resume section as ONE
    clean, colored card that mirrors the resume's own text as closely as
    possible: short un-bulleted lines are shown as entry headings (with
    any date range picked out as a small inline badge), longer
    un-bulleted lines as body text, and bulleted lines as a bullet list.
    This deliberately does not try to split the section into a separate
    card per line -- that guessing is what previously tore one education
    record (institution / degree / CGPA) into three disconnected boxes.
    Returns raw HTML so callers can compose several sections into a
    single st.markdown call (e.g. all inside one outer profile box)."""
    empty_html = f'<div style="color:#94A3B8; font-style:italic; font-size:14px;">{default_msg}</div>'

    if not section_content or "not clearly detected" in section_content.lower():
        return empty_html

    # ---- Pass 1: parse into structured (bullet/line) entries first,
    # instead of rendering straight from the raw lines. This lets us
    # detect and merge "orphan" wrapped lines before rendering -- see
    # the merge step below. ----
    entries = []  # [{"type": "bullet" | "line", "text": str}, ...]
    for raw_line in section_content.split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue
        is_bullet = bool(_BULLET_PREFIX_RE.match(stripped))
        line = _clean_resume_line(_BULLET_PREFIX_RE.sub("", stripped), strip_links)
        if not line:
            continue

        # A short (<=65 char), UN-bulleted line that immediately follows
        # a bullet is almost always that bullet's wrapped tail -- e.g. a
        # long project bullet that got split across two physical text
        # lines during PDF/DOCX extraction, where the second line (often
        # just its last few words) has no bullet character of its own.
        # Resumes place headings BEFORE their bullets, not after, so a
        # short line right after a bullet is far more likely to be a
        # wrap continuation than a genuine new heading. Without this
        # merge, that trailing fragment used to fall through to the
        # "short un-bulleted line -> bold heading" rule below and show
        # up as a few random bold words at the end of a project entry.
        if not is_bullet and entries and entries[-1]["type"] == "bullet" and len(line) <= 65:
            entries[-1]["text"] = entries[-1]["text"].rstrip() + " " + line
            continue

        entries.append({"type": "bullet" if is_bullet else "line", "text": line})

    if not entries:
        return empty_html

    # ---- Pass 2: render the merged entries ----
    rows_html = []
    for entry in entries:
        line = entry["text"]
        if entry["type"] == "bullet":
            rows_html.append(
                f'<div style="display:flex; gap:8px; margin:5px 0; font-size:13.5px; '
                f'color:#334155; line-height:1.55;">'
                f'<span style="color:{accent_color}; flex-shrink:0;">•</span><span>{line}</span></div>'
            )
        elif len(line) <= 65:
            top_margin = "0" if not rows_html else "12px"
            heading = _highlight_dates(line, accent_color, accent_bg)
            rows_html.append(
                f'<div style="font-size:14.5px; font-weight:700; color:#1E293B; '
                f'margin:{top_margin} 0 2px;">{heading}</div>'
            )
        else:
            rows_html.append(
                f'<div style="font-size:13.5px; color:#334155; line-height:1.55; margin:4px 0;">{line}</div>'
            )

    return f"""
    <div style="
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid {accent_color};
        border-radius: 12px;
        padding: 18px 20px 14px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    ">
        {''.join(rows_html)}
    </div>
    """


def render_structured_section(section_content: str, default_msg: str, accent_color: str = "#4F46E5",
                               accent_bg: str = "#EEF2F6", strip_links: bool = True):
    """Standalone-render wrapper around build_structured_section_html, kept
    for callers that want this section on its own (outside a composed box)."""
    st.markdown(
        _dedent_html(build_structured_section_html(section_content, default_msg, accent_color, accent_bg, strip_links)),
        unsafe_allow_html=True,
    )


def build_subsection_header_html(icon: str, label: str, color: str) -> str:
    """Builds (but does not render) the HTML for a colored mini-heading
    used to separate each block of the full candidate profile (Skills /
    Education / Experience / ...)."""
    return f"""
    <div style="
        display:flex; align-items:center; gap:8px;
        font-size:15px; font-weight:700; color:{color};
        margin: 22px 0 12px; padding-bottom:8px;
        border-bottom: 2px solid {color}33;
    ">
        <span style="font-size:16px;">{icon}</span><span>{label}</span>
    </div>
    """


def render_subsection_header(icon: str, label: str, color: str):
    """Standalone-render wrapper around build_subsection_header_html."""
    st.markdown(_dedent_html(build_subsection_header_html(icon, label, color)), unsafe_allow_html=True)


def render_summary_card(summary_text: str, accent_color: str = "#6366F1", accent_bg: str = "#EEF2FF"):
    """Renders the candidate's raw summary/objective section as a single
    highlighted card, preserving line breaks."""
    if not summary_text or "not clearly detected" in summary_text.lower():
        st.markdown(
            '<div style="color:#94A3B8; font-style:italic; font-size:14px;">'
            'No summary section clearly detected in this resume.</div>',
            unsafe_allow_html=True,
        )
        return

    paragraphs = [p.strip() for p in summary_text.split("\n") if p.strip()]
    body = "<br><br>".join(paragraphs)
    st.markdown(_dedent_html(f"""
    <div style="
        background: {accent_bg};
        border-left: 4px solid {accent_color};
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        font-size: 14px;
        line-height: 1.6;
        color: #1E293B;
    ">
        {body}
    </div>
    """), unsafe_allow_html=True)


# Palette cycled across skill categories so each group gets its own color.
_CATEGORY_PALETTE = [
    ("#6366F1", "#EEF2FF"),  # indigo
    ("#10B981", "#ECFDF5"),  # emerald
    ("#F59E0B", "#FFFBEB"),  # amber
    ("#3B82F6", "#EFF6FF"),  # blue
    ("#EC4899", "#FDF2F8"),  # pink
    ("#8B5CF6", "#F5F3FF"),  # purple
    ("#14B8A6", "#F0FDFA"),  # teal
    ("#F97316", "#FFF7ED"),  # orange
]

_SKILL_ACRONYMS = {
    "sql", "html", "css", "aws", "gcp", "api", "rest", "json", "xml", "ui",
    "ux", "ai", "ml", "nlp", "ci", "cd", "sdk", "saas", "php", "seo", "crm",
    "erp", "oop", "dbms", "http", "https", "aws", "azure", "sass",
}


def _display_skill_name(skill: str) -> str:
    """Best-effort pretty-print for a raw skill key (keeps recognized
    acronyms upper-cased, title-cases everything else)."""
    s = skill.strip()
    if s.lower() in _SKILL_ACRONYMS:
        return s.upper()
    if any(c.isupper() for c in s):
        return s
    return " ".join(w.capitalize() for w in s.replace("_", " ").split())


def build_skills_by_category_html(skills_by_category: dict,accent_color: str = "#49F343") -> str:
    """Builds (but does not render) the HTML for the candidate's own skills
    (independent of any JD) as color-coded chip groups, one color per
    category."""
    skills_by_category = {k: v for k, v in (skills_by_category or {}).items() if v}
    if not skills_by_category:
        return (
            '<div style="color:#94A3B8; font-style:italic; font-size:14px;">'
            'No recognizable skills found in this resume.</div>'
        )

    groups_html = []
    for i, (category, skills) in enumerate(skills_by_category.items()):
        color, bg = _CATEGORY_PALETTE[i % len(_CATEGORY_PALETTE)]
        chips = "".join(
            f'<span style="display:inline-block; background:{bg}; color:{color}; '
            f'padding:5px 12px; border-radius:20px; font-size:12.5px; font-weight:600; '
            f'margin:3px;">{_display_skill_name(s)}</span>'
            for s in skills
        )
        groups_html.append(f"""
        <div style="margin-bottom:14px;">
            <div style="font-size:11px; font-weight:700; text-transform:uppercase;
                        letter-spacing:0.5px; color:#000000; margin-bottom:6px;">
                {category} {chips}
            </div>
            
        </div>
        """)
    # return "".join(groups_html)
    return f"""
    <div style="
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid {accent_color};
        border-radius: 12px;
        padding: 18px 20px 14px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    ">
        {''.join(groups_html)} </div>"""


def render_skills_by_category(skills_by_category: dict):
    """Standalone-render wrapper around build_skills_by_category_html."""
    st.markdown(_dedent_html(build_skills_by_category_html(skills_by_category)), unsafe_allow_html=True)


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
    .candidate-detail-grid {
        display:grid; grid-template-columns:1fr 1fr; gap:0 28px; margin-top:8px;
    }
    @media (max-width:900px) {
        .candidate-detail-grid { grid-template-columns:1fr; }
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

        # Persist to the candidate store so the Recruiter Dashboard picks up
        # the resume score. Reuse an existing candidate_id (if the interview
        # was already taken) so this updates the same record instead of
        # creating a second one with only half the scores filled in.
        candidate_id = st.session_state.get("candidate_id")
        candidate_id = candidate_store.upsert_candidate(
            candidate_id, result.candidate.get("name", "Candidate"),
            {
                "resume_score": result.job_match_score,
                "resume_report": result.to_dict(),
            },
        )
        st.session_state.candidate_id = candidate_id

        st.success("Done — here's how your resume compares.")
    if run or result is not None:
        # ---- Candidate profile (what the tool read from your resume) ----
        st.markdown('<div class="section-title">Candidate Information</div>', unsafe_allow_html=True)

        name = result.candidate.get("name", "Unknown Candidate")
        initials = "".join([part[0].upper() for part in name.split() if part][:2])
        experience = f"{result.candidate['years_experience']} yrs" if result.candidate.get("years_experience") else "Not found"
        education = result.candidate.get("highest_education", "Not specified")
        email = result.candidate.get("email", "Not found")
        phone = result.candidate.get("phone", "Not found")
    
        sections_raw = result.sections or {}

        # ---- Left/right columns, composed into ONE html string and
        #      rendered with a single st.markdown call, so it all sits
        #      inside the same outer box as the name/avatar/stat row
        #      above -- not as separate boxes underneath it. Summary is
        #      intentionally excluded here per product decision. ----
        col1_html = "".join([
            build_subsection_header_html("🧠", "Skills", "#10B981"),
            build_skills_by_category_html(result.resume_skills_by_category),

            build_subsection_header_html("🎓", "Education", "#8B5CF6"),
            build_structured_section_html(
                sections_raw.get("education", ""),
                "No education section clearly detected in this resume.",
                accent_color="#8B5CF6", accent_bg="#F5F3FF",
            ),

            build_subsection_header_html("📜", "Certifications", "#EC4899"),
            build_structured_section_html(
                sections_raw.get("certifications", ""),
                "No certifications section clearly detected in this resume.",
                accent_color="#EC4899", accent_bg="#FDF2F8",
            ),
        ])

        col2_html = "".join([
            build_subsection_header_html("💼", "Experience", "#3B82F6"),
            build_structured_section_html(
                sections_raw.get("experience", ""),
                "No experience section clearly detected in this resume.",
                accent_color="#3B82F6", accent_bg="#EFF6FF",
            ),

            build_subsection_header_html("📁", "Projects", "#F59E0B"),
            build_structured_section_html(
                sections_raw.get("projects", ""),
                "No projects section clearly detected in this resume.",
                accent_color="#F59E0B", accent_bg="#FFFBEB",
            ),
        ])

        details_html = f"""
        <div class="candidate-detail-grid">
            <div>{col1_html}</div>
            <div>{col2_html}</div>
        </div>
        """

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
            <hr style="border:none; border-top:1px solid #E2E8F0; margin:20px 0 4px;">
            {details_html}
        </div>
        """
        st.markdown(_dedent_html(profile_html), unsafe_allow_html=True)
        # st.caption(
        #     "If any of this looks wrong or missing, an applicant tracking system "
        #     "will likely misread it too — worth fixing before you apply."
        # )

        # ---- Extracted text, section by section ----
        st.markdown('<div class="section-title">Section by Section Comparison</div>', unsafe_allow_html=True)
        # st.caption(
        #     "This is exactly what the parser pulled out of your file. If a section "
        #     "below looks empty, it likely means that heading wasn't clearly labeled "
        #     "in your resume — worth fixing, since an ATS will have the same trouble."
        # )

    
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

        # # ---- Section-wise recommendations ----
        # st.markdown('<div class="section-title">Section-wise recommendations</div>', unsafe_allow_html=True)
        # for rec in result.recommendations:
        #     flag = "🔴" if rec["needs_attention"] else "🟢"
        #     st.write(f"{flag} **{rec['section']}** — {rec['recommendation']}")

        # ---- Key scores ----
        st.markdown('<div class="section-title">Your scores</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="value">{result.resume_completeness["score"]}%</div><div class="label">Resume Completeness</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="value">{result.job_match_score}%</div><div class="label">Job Match Score</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="value">{result.resume_structure["score"]}%</div><div class="label">Resume Structure</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="value" style="font-size:18px">{result.interview_readiness["level"]}</div><div class="label">Interview Readiness</div></div>', unsafe_allow_html=True)

        st.info(result.interview_readiness["summary"])
        # st.caption(f"Hiring-style read: **{result.hiring_recommendation}**")

        # ---- Score breakdown ----
        st.markdown('<div class="section-title">Job Match Score breakdown</div>', unsafe_allow_html=True)
        # st.caption(
        #     "Skills 35% · Experience 20% · Education 10% · Resume↔JD Semantic Similarity 20% · Project Quality 15%"
        # )
        sb1, sb2, sb3, sb4, sb5 = st.columns(5)
        breakdown = calculate_job_match_breakdown(
            skills_score=result.skills_score,
            experience_score=result.experience_score,
            education_score=result.education_score,
            semantic_score=result.semantic_similarity["score"],
            project_quality_score=result.project_analysis["score"],
        )

        with sb1:
            st.write(f"🛠️ **Skills**: {breakdown['skills']:.1f}/35")
            st.progress(breakdown['skills'] / 35)

        with sb2:
            st.write(f"💼 **Experience**: {breakdown['experience']:.1f}/20")
            st.progress(breakdown['experience'] / 20)

        with sb3:
            st.write(f"🎓 **Education**: {breakdown['education']:.1f}/10")
            st.progress(breakdown['education'] / 10)

        with sb4:
            st.write(f"🧠 **Semantic Match**: {breakdown['semantic']:.1f}/20")
            st.progress(breakdown['semantic'] / 20)

        with sb5:
            st.write(f"📁 **Project Quality**: {breakdown['project_quality']:.1f}/15")
            st.progress(breakdown['project_quality'] / 15)
        # st.caption(f"Semantic similarity method: {result.semantic_similarity['method']}")

        

        # # ---- Resume completeness & structure ----
        # st.markdown('<div class="section-title">Resume completeness & structure</div>', unsafe_allow_html=True)
        # rc1, rc2 = st.columns(2)
        # with rc1:
        #     st.write(f"**Completeness — {result.resume_completeness['score']}%**")
        #     for f in result.resume_completeness["feedback"]:
        #         st.write(f"- {f}")
        # with rc2:
        #     st.write(f"**Structure — {result.resume_structure['score']}%**")
        #     for f in result.resume_structure["feedback"]:
        #         st.write(f"- {f}")

        # # ---- Project analysis ----
        # st.markdown('<div class="section-title">Project analysis</div>', unsafe_allow_html=True)
        # pa = result.project_analysis
        # st.write(
        #     f"Estimated projects: **{pa['project_count_estimate']}** · "
        #     f"Quantified impact: **{'Yes' if pa['has_quantified_impact'] else 'No'}** · "
        #     f"Relevant tech mentioned: **{', '.join(pa['relevant_tech_mentioned']) or 'None'}**"
        # )
        # for f in pa["feedback"]:
        #     st.write(f"- {f}")

        # # ---- Strengths & weaknesses ----
        # st.markdown('<div class="section-title">Strengths & weaknesses</div>', unsafe_allow_html=True)
        # sw1, sw2 = st.columns(2)
        # with sw1:
        #     st.write("**Strengths**")
        #     for s in result.strengths_weaknesses["strengths"]:
        #         st.success(s)
        # with sw2:
        #     st.write("**Weaknesses**")
        #     for w in result.strengths_weaknesses["weaknesses"]:
        #         st.error(w)

        

        # ---- Why this score (explainability) ----
        st.markdown('<div class="section-title">Why you got this score</div>', unsafe_allow_html=True)
        for key, item in result.explanation.items():
            label = key.replace("_", " ").capitalize()
            st.write(f"**{label}** — {item['value']}")

        # ---- Interview prep talking points ----
        st.markdown('<div class="section-title">Before you walk into the interview</div>', unsafe_allow_html=True)
        for s in result.interview_readiness["talking_points"]:
            st.write(f"- {s}")

        # with st.expander("Full raw output (JSON) — structured for downstream use"):
        #     st.json(result.to_dict())

        st.divider()
        if st.button("Continue to AI Interview →", type="primary"):
            st.session_state.nav_target = "Mock Interview"
            st.rerun()

    else:
        st.markdown('<p class="small-note">Add a resume and a job description above, then select "Check my resume".</p>', unsafe_allow_html=True)