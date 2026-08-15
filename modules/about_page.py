import streamlit as st


def _section_card(title: str, subtitle: str, body: str, accent: str = "#4F46E5"):
    st.markdown(
        f"""
        <div style="
            border: 1px solid #E2E8F0;
            border-radius: 18px;
            padding: 18px 20px;
            background: #FFFFFF;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
            height: 100%;
        ">
            <div style="display:flex; align-items:flex-start; gap:12px; margin-bottom:10px;">
                <div style="width:12px; height:12px; border-radius:999px; background:{accent}; margin-top:7px;"></div>
                <div>
                    <div style="font-size:18px; font-weight:700; color:#0F172A;">{title}</div>
                    <div style="font-size:12.5px; color:#64748B; margin-top:2px;">{subtitle}</div>
                </div>
            </div>
            <div style="font-size:14px; line-height:1.65; color:#334155;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render():
    st.title("ℹ️ About this app")
    st.caption("A section-by-section overview of the models, fallbacks, and responsibilities used across the platform.")

    st.markdown(
        """
        <div style="
            padding: 18px 20px;
            border-radius: 18px;
            background: linear-gradient(135deg, #EEF2FF 0%, #F8FAFC 55%, #ECFDF5 100%);
            border: 1px solid #E2E8F0;
            margin-bottom: 18px;
        ">
            <div style="font-size:16px; font-weight:700; color:#0F172A; margin-bottom:4px;">What this platform does</div>
            <div style="font-size:14px; color:#334155; line-height:1.6;">
                The app combines resume screening, an mock interview, interview scoring, and a learning roadmap into one flow.
                Each page uses a separate set of models or rules, and most features fall back gracefully if an API key or model package is missing.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Core sections")
    c1, c2 = st.columns(2)
    with c1:
        _section_card(
            "Resume Analysis",
            "ATS-style resume and job-description matching",
            """
            This page uses <b>spaCy en_core_web_sm</b> for entity extraction and resume parsing, then applies
            <b>sentence-transformers</b> with <b>all-MiniLM-L6-v2</b> for semantic matching between the resume and the job description.
            If embeddings are not available, it falls back to <b>TF-IDF cosine similarity</b> so the page still works.
            The engine also uses heuristics for section detection, completeness scoring, skill grouping, and candidate/contact extraction.
            """,
            accent="#4F46E5",
        )
    with c2:
        _section_card(
            "Mock Interview",
            "Live voice interview with webcam-based monitoring",
            """
            The interview page is built around <b>WebRTC</b> for browser audio/video capture.
            Speech-to-text uses only <b>SenseVoice</b> via <b>funasr</b> with
            model <b>iic/SenseVoiceSmall</b>.
            """,
            accent="#12B76A",
        )

    c3, c4 = st.columns(2)
    with c3:
        _section_card(
            "Interview Vision Checks",
            "Face, attention, and object detection during the interview",
            """
            Visual monitoring uses <b>OpenCV YuNet</b> face detection from <b>face_detection_yunet_2023mar.onnx</b>,
            <b>MediaPipe FaceMesh</b> for landmarks and gaze/head-pose signals, <b>MediaPipe Hands</b> for occlusion checks,
            and <b>YOLO26n</b> from <b>yolo26n.pt</b> for object detection. These signals drive the live attention and behavior panel.
            """,
            accent="#F59E0B",
        )
    with c4:
        _section_card(
            "Interview Scoring and Report",
            "LLM grading with a transparent fallback",
            """
            Answer scoring uses the <b>Google Gemini</b> API when <b>GEMINI_API_KEY</b> or <b>GOOGLE_API_KEY</b> is set.
            The evaluator scores <b>technical accuracy</b>, <b>communication</b>, <b>confidence</b>, <b>grammar</b>,
            <b>completeness</b>, and <b>relevance</b>. If Gemini is unavailable, the app switches to a deterministic heuristic scorer.
            The PDF report is generated with <b>ReportLab</b> from the stored resume and interview outputs.
            """,
            accent="#F04438",
        )

    st.subheader("Learning roadmap")
    st.markdown(
        """
        <div style="
            border: 1px solid #E2E8F0;
            border-radius: 18px;
            padding: 18px 20px;
            background: #FFFFFF;
        ">
            <div style="font-size:16px; font-weight:700; color:#0F172A; margin-bottom:6px;">Roadmap generation</div>
            <div style="font-size:14px; line-height:1.65; color:#334155;">
                The roadmap page combines the resume strengths with the interview weaknesses and uses <b>Gemini</b> when available to generate the plan.
                If Gemini is missing or fails, it falls back to a rules-based roadmap that still uses curated resources and the interview feedback stored in session state.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    