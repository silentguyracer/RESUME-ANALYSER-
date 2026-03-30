import streamlit as st
import google.generativeai as genai
import pymupdf
import os
from dotenv import load_dotenv
import re
import json

load_dotenv()

# Fetch API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY not found. Please set it in your `.env` file.")
    st.stop()

# Configure Google Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Structured prompt for consistent, parseable output
ANALYSIS_PROMPT = """You are an expert ATS (Applicant Tracking System) with deep understanding of tech roles, software engineering, data science, and other technical domains.

Evaluate the provided resume against the given job description. Return your analysis in the following **exact JSON format** (no markdown, no code fences, just raw JSON):

{
  "match_score": <integer 0-100>,
  "verdict": "<PASS or FAIL based on the score — PASS if score >= threshold, FAIL otherwise>",
  "summary": "<2-3 sentence overall assessment>",
  "strengths": ["<strength 1>", "<strength 2>", "..."],
  "weaknesses": ["<weakness 1>", "<weakness 2>", "..."],
  "missing_keywords": ["<keyword 1>", "<keyword 2>", "..."],
  "suggestions": ["<suggestion 1>", "<suggestion 2>", "..."],
  "experience_match": "<Strong / Moderate / Weak>",
  "skills_match": "<Strong / Moderate / Weak>",
  "education_match": "<Strong / Moderate / Weak>",
  "formatting_score": "<Good / Average / Poor>"
}

IMPORTANT RULES:
- The match_score MUST be an integer between 0 and 100.
- The threshold for PASS is {threshold}%. If match_score >= {threshold}, verdict is "PASS". Otherwise "FAIL".
- Be strict and realistic in scoring. Don't inflate the score.
- Provide at least 3 items in strengths, weaknesses, missing_keywords, and suggestions.
- Return ONLY valid JSON. No extra text, no markdown formatting.
"""


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pymupdf.open(stream=pdf_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text("text")
    return text


# Function to analyze resume against job description
def analyze_resume(resume_text, job_desc, threshold):
    prompt = ANALYSIS_PROMPT.format(threshold=threshold)
    full_prompt = f"{prompt}\n\nResume:\n{resume_text}\n\nJob Description:\n{job_desc}"
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(full_prompt)
    return response.text


def parse_analysis(raw_text):
    """Parse the JSON response from Gemini, handling potential formatting issues."""
    # Remove markdown code fences if present
    cleaned = raw_text.strip()
    cleaned = re.sub(r'^```json\s*', '', cleaned)
    cleaned = re.sub(r'^```\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                return None
    return None


# ─────────────────────────────────────────────
# Streamlit Page Config & Custom CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="ATS Resume Analyzer", page_icon="📄", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main { background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%); }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }

    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 2.5rem !important;
        text-align: center;
    }

    .subtitle {
        text-align: center;
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .pass-banner {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 24px 32px;
        border-radius: 16px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 176, 155, 0.3);
        animation: pulse 2s ease-in-out infinite;
    }

    .fail-banner {
        background: linear-gradient(135deg, #e53e3e, #fc466b);
        color: white;
        padding: 24px 32px;
        border-radius: 16px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(229, 62, 62, 0.3);
        animation: shake 0.5s ease-in-out;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }

    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }

    .score-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .score-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
    }

    .metric-label {
        color: #a0aec0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 4px;
    }

    .strong { color: #48bb78; }
    .moderate { color: #ecc94b; }
    .weak { color: #fc8181; }

    .tag {
        display: inline-block;
        background: rgba(102, 126, 234, 0.15);
        color: #a3bffa;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 4px;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }

    .tag-red {
        background: rgba(252, 70, 107, 0.15);
        color: #feb2b2;
        border: 1px solid rgba(252, 70, 107, 0.3);
    }

    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }

    .list-item {
        color: #cbd5e0;
        padding: 8px 0;
        padding-left: 20px;
        position: relative;
        line-height: 1.6;
    }

    .list-item::before {
        content: '▸';
        position: absolute;
        left: 0;
        color: #667eea;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 40px !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
    }

    .stFileUploader {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        border: 1px dashed rgba(102, 126, 234, 0.3);
        padding: 10px;
    }

    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }

    .stSlider > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }

    .threshold-info {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 16px;
        color: #a3bffa;
        font-size: 0.9rem;
        margin-top: 8px;
    }

    .summary-box {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #667eea;
        padding: 16px 20px;
        border-radius: 0 12px 12px 0;
        color: #e2e8f0;
        font-size: 1rem;
        line-height: 1.7;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# App Header
# ─────────────────────────────────────────────
st.markdown("<h1>📄 ATS Resume Analyzer</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload your resume • Set your pass threshold • Get instant AI-powered feedback</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar — Settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    threshold = st.slider(
        "Pass/Fail Threshold (%)",
        min_value=30,
        max_value=90,
        value=60,
        step=5,
        help="Resumes scoring at or above this threshold will PASS"
    )
    st.markdown(f"""
    <div class="threshold-info">
        📌 Resumes scoring <b>≥ {threshold}%</b> will <b style="color:#48bb78;">PASS</b><br>
        📌 Resumes scoring <b>&lt; {threshold}%</b> will <b style="color:#fc8181;">FAIL</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📖 How it works")
    st.markdown("""
    1. **Upload** your resume (PDF)
    2. **Paste** the target job description
    3. **Set** your desired pass threshold
    4. **Click** Analyze to get your report
    """)
    st.markdown("---")
    st.markdown(
        "<p style='color:#718096; font-size:0.8rem; text-align:center;'>Powered by Google Gemini AI</p>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# Input Section
# ─────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📎 Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF format only)",
        type=["pdf"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        st.success(f"✅ Uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

with col2:
    st.markdown("### 📝 Job Description")
    job_description = st.text_area(
        "Enter Job Description",
        placeholder="Paste the full job description here...",
        height=200,
        label_visibility="collapsed"
    )

# Centered Analyze Button
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    analyze_button = st.button("🔍 Analyze Resume", use_container_width=True)

# ─────────────────────────────────────────────
# Validation & Analysis
# ─────────────────────────────────────────────
if analyze_button:
    if not uploaded_file:
        st.warning("⚠️ Please upload a resume PDF.")
    elif not job_description.strip():
        st.warning("⚠️ Please enter a job description.")
    else:
        # Extract text from PDF
        with st.spinner("📑 Extracting text from resume..."):
            resume_text = extract_text_from_pdf(uploaded_file)

        if not resume_text or len(resume_text.strip()) < 50:
            st.error("❌ Could not extract enough text from the PDF. Please make sure it's not a scanned image.")
        else:
            # Analyze the resume
            with st.spinner("🤖 Analyzing your resume against the job description..."):
                raw_analysis = analyze_resume(resume_text, job_description, threshold)
                analysis = parse_analysis(raw_analysis)

            if analysis is None:
                st.error("❌ Failed to parse AI response. Showing raw output:")
                st.code(raw_analysis)
            else:
                # ── Results Section ──
                st.markdown("---")
                score = analysis.get("match_score", 0)
                verdict = "PASS" if score >= threshold else "FAIL"

                # ── Pass / Fail Banner ──
                if verdict == "PASS":
                    st.markdown(f"""
                    <div class="pass-banner">
                        ✅ PASS — Your resume scored {score}% (Threshold: {threshold}%)
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="fail-banner">
                        ❌ FAIL — Your resume scored {score}% (Threshold: {threshold}%)
                    </div>
                    """, unsafe_allow_html=True)

                # ── Score + Sub-Metrics Row ──
                st.markdown('<div class="section-header">📊 Score Breakdown</div>', unsafe_allow_html=True)

                m1, m2, m3, m4 = st.columns(4)
                metrics = [
                    ("🎯 Match Score", f"{score}%", "strong" if score >= 70 else "moderate" if score >= 50 else "weak"),
                    ("💼 Experience", analysis.get("experience_match", "N/A"),
                     "strong" if analysis.get("experience_match") == "Strong" else "moderate" if analysis.get("experience_match") == "Moderate" else "weak"),
                    ("🛠️ Skills", analysis.get("skills_match", "N/A"),
                     "strong" if analysis.get("skills_match") == "Strong" else "moderate" if analysis.get("skills_match") == "Moderate" else "weak"),
                    ("🎓 Education", analysis.get("education_match", "N/A"),
                     "strong" if analysis.get("education_match") == "Strong" else "moderate" if analysis.get("education_match") == "Moderate" else "weak"),
                ]

                for col, (label, value, css_class) in zip([m1, m2, m3, m4], metrics):
                    with col:
                        st.markdown(f"""
                        <div class="score-card">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value {css_class}">{value}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # ── Progress Bar ──
                st.progress(score / 100)

                # ── Formatting Score ──
                fmt = analysis.get("formatting_score", "N/A")
                fmt_class = "strong" if fmt == "Good" else "moderate" if fmt == "Average" else "weak"
                st.markdown(f"""
                <div class="score-card">
                    <div class="metric-label">📝 Resume Formatting</div>
                    <div class="metric-value {fmt_class}">{fmt}</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Summary ──
                summary = analysis.get("summary", "")
                if summary:
                    st.markdown('<div class="section-header">💡 Summary</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

                # ── Strengths & Weaknesses ──
                col_sw1, col_sw2 = st.columns(2)
                with col_sw1:
                    st.markdown('<div class="section-header">✅ Strengths</div>', unsafe_allow_html=True)
                    for item in analysis.get("strengths", []):
                        st.markdown(f'<div class="list-item">{item}</div>', unsafe_allow_html=True)

                with col_sw2:
                    st.markdown('<div class="section-header">⚠️ Weaknesses</div>', unsafe_allow_html=True)
                    for item in analysis.get("weaknesses", []):
                        st.markdown(f'<div class="list-item">{item}</div>', unsafe_allow_html=True)

                # ── Missing Keywords ──
                missing_kw = analysis.get("missing_keywords", [])
                if missing_kw:
                    st.markdown('<div class="section-header">🔑 Missing Keywords</div>', unsafe_allow_html=True)
                    tags_html = " ".join([f'<span class="tag tag-red">{kw}</span>' for kw in missing_kw])
                    st.markdown(tags_html, unsafe_allow_html=True)

                # ── Suggestions ──
                suggestions = analysis.get("suggestions", [])
                if suggestions:
                    st.markdown('<div class="section-header">💡 Improvement Suggestions</div>', unsafe_allow_html=True)
                    for i, sug in enumerate(suggestions, 1):
                        st.markdown(f'<div class="list-item"><b>{i}.</b> {sug}</div>', unsafe_allow_html=True)

                # ── Download Report ──
                st.markdown("<br>", unsafe_allow_html=True)
                report_lines = []
                report_lines.append("=" * 60)
                report_lines.append("    ATS RESUME ANALYSIS REPORT")
                report_lines.append("=" * 60)
                report_lines.append("")
                report_lines.append(f"  VERDICT:    {verdict}")
                report_lines.append(f"  SCORE:      {score}%")
                report_lines.append(f"  THRESHOLD:  {threshold}%")
                report_lines.append("")
                report_lines.append("-" * 60)
                report_lines.append("SUMMARY")
                report_lines.append("-" * 60)
                report_lines.append(summary)
                report_lines.append("")
                report_lines.append("-" * 60)
                report_lines.append("SUB-SCORES")
                report_lines.append("-" * 60)
                report_lines.append(f"  Experience Match:  {analysis.get('experience_match', 'N/A')}")
                report_lines.append(f"  Skills Match:      {analysis.get('skills_match', 'N/A')}")
                report_lines.append(f"  Education Match:   {analysis.get('education_match', 'N/A')}")
                report_lines.append(f"  Formatting:        {analysis.get('formatting_score', 'N/A')}")
                report_lines.append("")
                report_lines.append("-" * 60)
                report_lines.append("STRENGTHS")
                report_lines.append("-" * 60)
                for s in analysis.get("strengths", []):
                    report_lines.append(f"  ✓ {s}")
                report_lines.append("")
                report_lines.append("-" * 60)
                report_lines.append("WEAKNESSES")
                report_lines.append("-" * 60)
                for w in analysis.get("weaknesses", []):
                    report_lines.append(f"  ✗ {w}")
                report_lines.append("")
                report_lines.append("-" * 60)
                report_lines.append("MISSING KEYWORDS")
                report_lines.append("-" * 60)
                for kw in analysis.get("missing_keywords", []):
                    report_lines.append(f"  • {kw}")
                report_lines.append("")
                report_lines.append("-" * 60)
                report_lines.append("SUGGESTIONS")
                report_lines.append("-" * 60)
                for i, sug in enumerate(analysis.get("suggestions", []), 1):
                    report_lines.append(f"  {i}. {sug}")
                report_lines.append("")
                report_lines.append("=" * 60)

                report_text = "\n".join(report_lines)

                col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 1])
                with col_dl2:
                    st.download_button(
                        label="📥 Download Full Report",
                        data=report_text,
                        file_name="resume_analysis_report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
