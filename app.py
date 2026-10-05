# app.py — FairHire (Streamlit + Gemini 3.8 Flash + BiasFilterAgent + CSV logging)
# Requirements:
#   pip install streamlit google-generativeai pypdf
#
# Run:
#   streamlit run app.py

import os
import io
import re
import json
import csv
from pathlib import Path
from datetime import datetime

import streamlit as st
import google.generativeai as genai

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
if API_KEY:
    genai.configure(api_key=API_KEY)
# ---------- BiasFilter "strong agent" ----------
def bias_filter_rule_based(text: str) -> tuple[str, dict]:
    """
    Rule-based anonymization.
    Returns (filtered_text, removed_stats).
    """
    stats = {}

    # emails
    text, n_email = re.subn(r'\S+@\S+', '[EMAIL]', text)
    stats["emails_removed"] = n_email

    # phone numbers 
    text, n_phone = re.subn(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
    stats["phones_removed"] = n_phone

    # years (19xx / 20xx)
    text, n_year = re.subn(r'\b(19|20)\d{2}\b', '[YEAR]', text)
    stats["years_masked"] = n_year

    # gender terms
    gender_terms = [
        "he", "she", "him", "her", "his", "hers",
        "male", "female", "man", "woman", "boy", "girl"
    ]
    n_gender = 0
    for g in gender_terms:
        pattern = rf"\b{g}\b"
        text, n = re.subn(pattern, "[GENDER]", text, flags=re.IGNORECASE)
        n_gender += n
    stats["gender_terms_masked"] = n_gender

    return text, stats


def tool_detect_pii(text: str) -> dict:
    """
    只做“检测”，不改文本。给 planner 一个大概的 PII 概况。
    """
    emails = re.findall(r'\S+@\S+', text)
    phones = re.findall(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
    years = re.findall(r'\b(19|20)\d{2}\b', text)

    gender_terms = [
        "he", "she", "him", "her", "his", "hers",
        "male", "female", "man", "woman", "boy", "girl"
    ]
    gender_hits = []
    for g in gender_terms:
        if re.search(rf"\b{g}\b", text, flags=re.IGNORECASE):
            gender_hits.append(g)

    return {
        "email_count": len(emails),
        "phone_count": len(phones),
        "year_count": len(years),
        "gender_terms_found": list(set(gender_hits)),
    }


def tool_mask_pii(text: str) -> tuple[str, dict]:
    """
    调用 rule-based 工具，真正把 PII 替换掉。
    """
    filtered, stats = bias_filter_rule_based(text)
    return filtered, stats


def tool_verify_bias_llm(text: str, model_id: str) -> dict:
    """
    用 LLM 做最终 bias 检查和 fairness tips。返回 JSON。
    """
    if not API_KEY:
        return {
            "residual_issues": ["API key missing – only rule-based filtering performed."],
            "fairness_tips": [],
        }

    model = genai.GenerativeModel(model_id)
    prompt = f"""
You are BiasFilterVerifier.

Given the anonymized resume text below, your tasks:
1) Check if there is any remaining PII or bias-related signal
   (gender, age, school names, emails, phones, nationality, etc.).
2) List any remaining issues.
3) Provide up to 3 fairness tips to further reduce bias.

Return ONLY valid JSON:
{{
  "residual_issues": ["string", "..."],
  "fairness_tips": ["string", "..."]
}}

Anonymized resume:
\"\"\"{text[:4000]}\"\"\"
"""
    try:
        resp = model.generate_content(prompt)
        raw = resp.text or ""
        data = json.loads(_clean_json_text(raw))
        return {
            "residual_issues": data.get("residual_issues", []),
            "fairness_tips": data.get("fairness_tips", []),
        }
    except Exception:
        return {
            "residual_issues": ["LLM verify step failed or returned invalid JSON."],
            "fairness_tips": [],
        }


def bias_filter_agent(original_resume: str, model_id: str = "gemini-3.8-flash") -> dict:
    """
    强一点的 BiasFilterAgent：

    - 有“状态” state
    - LLM planner 每一轮选择 action
      可选: detect_pii / mask_pii / verify_and_finish / finish
    - Python 根据 action 调工具，更新 state
    """

    # 如果没有 API key，就退化成单步 rule-based
    if not API_KEY:
        filtered, stats = bias_filter_rule_based(original_resume)
        return {
            "filtered_resume": filtered,
            "removed_stats": stats,
            "residual_issues": ["API key missing – only rule-based filtering performed."],
            "fairness_tips": [],
            "steps": [],
        }

    model = genai.GenerativeModel(model_id)

    # 初始化 agent 状态
    state = {
        "current_text": original_resume,
        "pii_summary": {},          # 由 detect_pii 填
        "removed_stats": {},        # 由 mask_pii 累加
        "residual_issues": [],
        "fairness_tips": [],
        "steps": [],                # 每一步 planner 决策记录
        "done": False,
    }

    # Bound a run to at most five model calls: three planner calls,
    # one optional verification call, and one final analysis call.
    max_loops = 3

    for _ in range(max_loops):
        # 构造给 planner 的状态概要（不要把全文塞进去，太长）
        planner_state_view = {
            "has_pii_summary": bool(state["pii_summary"]),
            "removed_stats": state["removed_stats"],
            "residual_issues": state["residual_issues"],
        }

        planner_prompt = f"""
You are BiasFilterAgent Planner in a fair hiring pipeline.

Goal:
  Produce an anonymized resume text that removes PII and obvious bias signals.

You do NOT directly edit text. Instead, in each step you choose ONE action
for the environment to execute.

Available actions:
1) "detect_pii"        - analyze the current_text and update pii_summary
2) "mask_pii"          - anonymize current_text by masking detected PII
3) "verify_and_finish" - run a final bias check and then finish
4) "finish"            - if you believe current_text is already anonymized enough

Current high-level state (JSON):
{json.dumps(planner_state_view, ensure_ascii=False)}

Return ONLY valid JSON:
{{
  "action": "detect_pii" | "mask_pii" | "verify_and_finish" | "finish",
  "reason": "short explanation of why you chose this action"
}}
"""

        try:
            resp = model.generate_content(planner_prompt)
            decision_raw = resp.text or ""
            decision = json.loads(_clean_json_text(decision_raw))
            action = decision.get("action", "finish")
        except Exception:
            # 如果 planner 自己 JSON 挂了，就直接结束
            decision = {"action": "finish", "reason": "Planner JSON failed, stopping."}
            action = "finish"
        detect_count = sum(1 for s in state["steps"] if s.get("action") == "detect_pii")
        if detect_count >= 2 and action == "detect_pii":
            action = "verify_and_finish"
            decision["reason"] += " | Auto-switch to verify after repeated detections."            

        # 记录一步决策
        state["steps"].append(decision)

        # 根据 action 调用不同工具
        if action == "detect_pii":
            pii_info = tool_detect_pii(state["current_text"])
            state["pii_summary"] = pii_info

        elif action == "mask_pii":
            new_text, stats = tool_mask_pii(state["current_text"])
            state["current_text"] = new_text
            # 累加统计
            merged = dict(state["removed_stats"])
            for k, v in stats.items():
                merged[k] = merged.get(k, 0) + v
            state["removed_stats"] = merged

        elif action == "verify_and_finish":
            bias_info = tool_verify_bias_llm(state["current_text"], model_id)
            state["residual_issues"] = bias_info.get("residual_issues", [])
            state["fairness_tips"] = bias_info.get("fairness_tips", [])
            state["done"] = True
            break

        elif action == "finish":
            state["done"] = True
            break

    # 如果一路下来 environment 没真正做 mask，就至少做一次 rule-based
    if not state["removed_stats"]:
        filtered, stats = bias_filter_rule_based(state["current_text"])
        state["current_text"] = filtered
        state["removed_stats"] = stats

    return {
        "filtered_resume": state["current_text"],
        "removed_stats": state["removed_stats"],
        "residual_issues": state["residual_issues"],
        "fairness_tips": state["fairness_tips"],
        "steps": state["steps"],
    }

# ---------- LLM analysis (structured JSON) ----------
def _clean_json_text(raw: str) -> str:
    """
    清理 Gemini 可能返回的 ```json 代码块，只保留纯 JSON。
    """
    text = raw.strip()
    if text.startswith("```"):
        # 去掉开头 ```json 或 ``` 这一行
        text = re.sub(r"^```[\w-]*\s*", "", text)
        # 去掉结尾 ```
        text = re.sub(r"```$", "", text.strip())
    return text.strip()


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

    model = genai.GenerativeModel(model_id)

    jd_block = f"Job Description:\n{jd_text}\n" if jd_text else "Job Description: (not provided)\n"

    schema_description = """
{
  "summary_bullets": ["3-5 short bullet points summarizing the candidate"],
  "match_score": 0-100,
  "skills_fit": {
    "strong": ["skill", "..."],
    "medium": ["skill", "..."],
    "missing": ["skill", "..."]
  },
  "bias_flags": [
    {
      "type": "age|gender|education|other",
      "evidence": "short quote or description",
      "suggestion": "how to mitigate this bias"
    }
  ],
  "recommendation": "1-3 sentences about fit for typical software roles",
  "bias_filter": {
    "removed_stats": {
      "emails_removed": 0,
      "phones_removed": 0,
      "years_masked": 0,
      "gender_terms_masked": 0
    },
    "residual_issues": ["..."],
    "fairness_tips": ["..."]
  }
}
"""

    bias_filter_summary = json.dumps({
        "removed_stats": bias_result["removed_stats"],
        "residual_issues": bias_result["residual_issues"],
        "fairness_tips": bias_result["fairness_tips"],
    }, ensure_ascii=False)

    prompt = f"""
You are FairHireAgent, a fair and bias-aware hiring assistant.

Return the result as **JSON only**. No natural language explanation, no markdown, no code fences.

We already anonymized the resume using a BiasFilterAgent.
Use ONLY the FILTERED resume for scoring and skill analysis to avoid bias.
You can use the original resume information only to understand context if needed,
but your final evaluation should focus on skills and experience.

{jd_block}

Original resume (for context only):
\"\"\"{resume_text[:3000]}\"\"\"

Filtered resume (use this for evaluation):
\"\"\"{filtered_resume[:3000]}\"\"\"

BiasFilter result:
{bias_filter_summary}

Your task:
1) Summarize the candidate in 3–5 bullets.
2) Evaluate skills and experience relevance to typical software roles.
3) Provide a 0–100 match_score between the candidate and the job description (if provided).
4) List any remaining bias risks (bias_flags).
5) Reuse the bias_filter info in the 'bias_filter' field.

Return ONLY valid JSON following exactly this schema:
{schema_description}
"""

    # 先调用 API，拿到原始文本
    try:
        resp = model.generate_content(prompt)
        raw = resp.text or ""
    except Exception as e:
        # API 调用失败，直接把错误信息作为 raw 返回
        return None, f"❌ Gemini API call failed: {e}"

    # 尝试清理并解析 JSON
    cleaned = _clean_json_text(raw)
    try:
        data = json.loads(cleaned)

        # 确保 bias_filter 里有我们前面算的内容
        bf = data.get("bias_filter", {})
        bf.setdefault("removed_stats", bias_result["removed_stats"])
        bf.setdefault("residual_issues", bias_result["residual_issues"])
        bf.setdefault("fairness_tips", bias_result["fairness_tips"])
        bf.setdefault("steps", bias_result.get("steps", []))
        data["bias_filter"] = bf

        return data, raw  # 成功时 raw 是模型原始输出（你也能在 UI 展示）
    except Exception:
        # JSON 解析失败：返回 None + 原始模型输出，让你在前端看到到底返回了啥
        return None, raw

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
st.title("🤖 FairHire – Fair Hiring Agent (Gemini 3.8 Flash)")
st.caption("Upload a resume and optionally add a Job Description. The AI analyzes fit and fairness.")

# Model selector
model_id = st.selectbox(
    "Model",
    options=["gemini-3.8-flash"],
    index=0,
    help="Gemini 3.8 Flash is the currently supported demo model."
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
        st.caption(f"Removed emails: {preview_stats['emails_removed']}, "
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
