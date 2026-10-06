# app.py — FairHire (Streamlit + Gemini + BiasFilterAgent + CSV logging)
# Requirements:
#   pip install streamlit google-genai pypdf
#
# Run:
#   streamlit run app.py

import os
import io
import json
import csv
from pathlib import Path
from datetime import datetime

import streamlit as st
from google import genai
from fairhire_core import (
    analysis_schema,
    bias_filter_rule_based,
    run_bias_filter_agent,
    validate_analysis,
)

# ---------- PDF parsing ----------
try:
    import pypdf
    HAS_PYPDF = True
except Exception:
    HAS_PYPDF = False

# ---------- Paths ----------
LOG_PATH = Path("fairhire_runs.csv")

# ---------- API key loading ----------
def get_api_key() -> str | None:
    # Try common env var names first, then Streamlit secrets
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            key = None
    return key

API_KEY = get_api_key()


def generate_text(prompt: str, model_id: str) -> str:
    """Call the current Gemini Interactions API without retaining an interaction."""
    with genai.Client(api_key=API_KEY) as client:
        interaction = client.interactions.create(
            model=model_id, input=prompt, store=False, timeout=30
        )
    return interaction.output_text or ""


# ---------- BiasFilter verification ----------
def bias_filter_agent(original_resume: str, model_id: str = "gemini-3.8-flash") -> dict:
    return run_bias_filter_agent(
        original_resume, model_id, generate_text if API_KEY else None
    )


# ---------- LLM analysis (structured JSON) ----------
def analyze_resume(
    resume_text: str,
    jd_text: str = "",
    model_id: str = "gemini-3.8-flash"
) -> tuple[dict | None, str | None]:
    """
    Main analysis:
    1) Run BiasFilterAgent
    2) Ask LLM to produce structured JSON summary / score / skills / bias flags
    Returns: (structured_dict or None, raw_model_output)
    """
    if not API_KEY:
        return None, "❌ Missing API key. Set GEMINI_API_KEY or GOOGLE_API_KEY."

    # --- Step 1: BiasFilterAgent ---
    bias_result = bias_filter_agent(resume_text, model_id=model_id)
    filtered_resume = bias_result["filtered_resume"]

    jd_block = f"Job Description:\n{jd_text}\n" if jd_text else "Job Description: (not provided)\n"

    schema_description = analysis_schema()

    bias_filter_summary = json.dumps({
        "removed_stats": bias_result["removed_stats"],
        "residual_issues": bias_result["residual_issues"],
        "fairness_tips": bias_result["fairness_tips"],
    }, ensure_ascii=False)

    prompt = f"""
You are FairHireAgent, a fair and bias-aware hiring assistant.

Return the result as **JSON only**. No natural language explanation, no markdown, no code fences.

The resume was partially de-identified by deterministic rules.
Use ONLY the FILTERED resume for scoring and skill analysis.
Do not infer protected attributes or make a hiring decision.

{jd_block}

Filtered resume (use this for evaluation):
\"\"\"{filtered_resume[:3000]}\"\"\"

BiasFilter result:
{bias_filter_summary}

Your task:
1) Summarize the candidate in 3–5 bullets.
2) Evaluate skills and experience relevance to typical software roles.
3) Provide a 0–100 match_score between the candidate and the job description (if provided).
4) List any remaining bias risks (bias_flags).
5) Do not repeat the bias_filter data; the application attaches it separately.

Return ONLY valid JSON following exactly this schema:
{schema_description}
"""

    # 先调用 API，拿到原始文本
    try:
        raw = generate_text(prompt, model_id)
    except Exception as e:
        # API 调用失败，直接把错误信息作为 raw 返回
        return None, f"❌ Gemini API call failed: {e}"

    try:
        return validate_analysis(raw, bias_result), raw
    except Exception:
        return None, "Model output did not match the required JSON schema."

# ---------- File helpers ----------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    if not HAS_PYPDF:
        return ""
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages).strip()

def read_uploaded_file(uploaded) -> str:
    if uploaded is None:
        return ""
    data = uploaded.read()
    if uploaded.name.lower().endswith(".pdf"):
        return extract_text_from_pdf(data)
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

# ---------- CSV logging ----------
def log_run(model_id: str, resume_text: str, jd_text: str, result: dict | None):
    """
    Append one analysis record to CSV.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 要写入的基本字段
    timestamp = datetime.now().isoformat(timespec="seconds")
    resume_len = len(resume_text)
    jd_len = len(jd_text)

    if result:
        match_score = result.get("match_score", "")
        bias_flags = result.get("bias_flags", [])
        bias_flag_count = len(bias_flags)
        removed_stats = result.get("bias_filter", {}).get("removed_stats", {})
    else:
        match_score = ""
        bias_flag_count = ""
        removed_stats = {}

    row = {
        "timestamp": timestamp,
        "model_id": model_id,
        "resume_chars": resume_len,
        "jd_chars": jd_len,
        "match_score": match_score,
        "bias_flag_count": bias_flag_count,
        "emails_removed": removed_stats.get("emails_removed", 0),
        "phones_removed": removed_stats.get("phones_removed", 0),
        "years_masked": removed_stats.get("years_masked", 0),
        "gender_terms_masked": removed_stats.get("gender_terms_masked", 0),
    }

    file_exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_history(limit: int = 5):
    if not LOG_PATH.exists():
        return []
    with LOG_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # 返回最近几条，按时间倒序
    return rows[-limit:][::-1]

# ---------- Session state defaults ----------
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""

# ---------- UI ----------
st.set_page_config(page_title="FairHire – Fair Hiring Agent", layout="centered")
st.title("🤖 FairHire – Fair Hiring Agent (Gemini)")
st.caption("Upload a resume and optionally add a Job Description. The AI analyzes fit and fairness.")

# Model selector
model_id = st.selectbox(
    "Model",
    options=["gemini-3.8-flash", "gemini-3.5-flash-lite"],
    index=0,
    help="Gemini 3.8 Flash is the default; Flash-Lite is a lower-latency alternative."
)

# Upload
uploaded = st.file_uploader("📄 Upload resume (.txt, .pdf)", type=["txt", "pdf"])

if uploaded is not None:
    raw_text = read_uploaded_file(uploaded)
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("↪️ Use extracted resume text"):
            st.session_state.resume_text = raw_text
            st.success("Filled resume text from uploaded file.")
            st.rerun()
    with col2:
        if uploaded.name.lower().endswith(".pdf") and not HAS_PYPDF:
            st.warning("PDF parsing requires `pypdf`. Install:  pip install pypdf")

# Always render BOTH inputs
st.text_area(
    "Resume text",
    key="resume_text",
    height=220,
    help="Paste resume text or click the button above to fill from the uploaded file."
)
st.text_area(
    "Job description (optional)",
    key="jd_text",
    height=160
)

# Optional: 展示 Bias-filtered 预览
if st.session_state.resume_text.strip():
    preview_filtered, preview_stats = bias_filter_rule_based(st.session_state.resume_text)
    with st.expander("👀 Bias-filtered resume preview (rule-based)", expanded=False):
        st.text_area(
            "Filtered (rule-based preview only)",
            value=preview_filtered,
            height=180,
            disabled=True
        )
        st.caption(f"Removed names: {preview_stats['names_removed']}, "
                   f"emails: {preview_stats['emails_removed']}, "
                   f"phones: {preview_stats['phones_removed']}, "
                   f"years: {preview_stats['years_masked']}, "
                   f"gender terms: {preview_stats['gender_terms_masked']}")

# Analyze
if st.button("🔍 Analyze"):
    resume_text = st.session_state.resume_text.strip()
    jd_text = st.session_state.jd_text.strip()

    if not API_KEY:
        st.error("No API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY.")
    elif not resume_text:
        st.warning("Please provide resume text (upload or paste).")
    else:
        with st.spinner(f"Analyzing with {model_id}..."):
            structured, raw = analyze_resume(resume_text, jd_text, model_id=model_id)

        # 写日志（无论 structured 成功与否）
        log_run(model_id, resume_text, jd_text, structured)

        if structured is None:
            st.error("Analysis failed.")
            st.text_area("Raw model output", value=str(raw), height=200)
        else:
            st.success("Analysis complete ✅")

            # ---- Structured UI ----
            st.subheader("🎯 Match score")
            st.metric("Resume–JD match (0–100)", structured.get("match_score", "N/A"))

            st.subheader("📋 Candidate summary")
            for bullet in structured.get("summary_bullets", []):
                st.markdown(f"- {bullet}")

            st.subheader("🧠 Skills fit")
            skills = structured.get("skills_fit", {})
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("**Strong**")
                for s in skills.get("strong", []):
                    st.markdown(f"- {s}")
            with col_b:
                st.markdown("**Medium**")
                for s in skills.get("medium", []):
                    st.markdown(f"- {s}")
            with col_c:
                st.markdown("**Missing**")
                for s in skills.get("missing", []):
                    st.markdown(f"- {s}")

            st.subheader("⚖️ Fairness & Bias Audit")
            bias_flags = structured.get("bias_flags", [])
            if not bias_flags:
                st.write("No obvious bias indicators detected in this resume/JD pair.")
            else:
                for flag in bias_flags:
                    st.markdown(f"- **Type**: {flag.get('type', 'unknown')}")
                    st.markdown(f"  - Evidence: {flag.get('evidence', '')}")
                    st.markdown(f"  - Suggestion: {flag.get('suggestion', '')}")

            bf = structured.get("bias_filter", {})
            with st.expander("BiasFilterAgent details"):
                st.json(bf)

            st.subheader("📝 Overall recommendation")
            st.write(structured.get("recommendation", ""))

            with st.expander("Raw JSON from model"):
                st.json(structured)

# History section
st.subheader("📊 Recent analyses (CSV log)")
history_rows = load_history(limit=5)
if history_rows:
    st.table(history_rows)
else:
    st.caption("No analyses logged yet. Run an analysis to start logging.")

# Footer tip
if not API_KEY:
    st.info(
        "Set your key in PowerShell:\n"
        '  [Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "your_key", "User")\n'
        "Restart terminal and run: streamlit run app.py"
    )
