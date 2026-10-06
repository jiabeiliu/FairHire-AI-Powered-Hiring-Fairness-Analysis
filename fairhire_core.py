"""Deterministic privacy filtering and validation for the FairHire demo."""

import json
import re
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class VerificationResult(StrictModel):
    residual_issues: list[str]
    fairness_tips: list[str] = Field(max_length=3)


class SkillsFit(StrictModel):
    strong: list[str]
    medium: list[str]
    missing: list[str]


class BiasFlag(StrictModel):
    type: Literal["age", "gender", "education", "other"]
    evidence: str
    suggestion: str


class AnalysisResult(StrictModel):
    summary_bullets: list[str] = Field(min_length=1, max_length=5)
    match_score: int = Field(ge=0, le=100)
    skills_fit: SkillsFit
    bias_flags: list[BiasFlag]
    recommendation: str = Field(min_length=1)


def clean_json_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[\w-]*\s*", "", text)
        text = re.sub(r"```$", "", text.strip())
    return text.strip()


def _mask_name(text: str) -> tuple[str, int]:
    """Mask explicit name labels and a likely resume-header name.

    This intentionally avoids broad NER guesses that could erase skills or
    job titles. Unusual name layouts still require human review.
    """
    text, labeled_count = re.subn(
        r"(?im)^(\s*(?:full\s+name|name|姓名)\s*[:：]\s*)[^\r\n]+",
        r"\1[NAME]",
        text,
    )
    lines = text.splitlines(keepends=True)
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is None:
        return text, labeled_count

    candidate = lines[first].strip()
    words = candidate.split()
    capitalized_words = all(
        re.fullmatch(r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)*", word) for word in words
    )
    not_job_title = not any(word.lower() in {
        "engineer", "developer", "analyst", "manager", "scientist",
        "designer", "resume", "curriculum", "vitae", "summary",
    } for word in words)
    looks_like_name = (
        2 <= len(words) <= 4 and capitalized_words and not_job_title
    ) or bool(re.fullmatch(r"[\u3400-\u9fff]{2,4}", candidate))
    contact_nearby = any(
        re.search(r"@|\b(?:email|phone|tel|contact)\b|\b\d{3}[-.\s]?\d{3}", line, re.I)
        for line in lines[first + 1:first + 4]
    )
    if looks_like_name and contact_nearby:
        lines[first] = lines[first].replace(candidate, "[NAME]", 1)
        return "".join(lines), labeled_count + 1
    return text, labeled_count


def bias_filter_rule_based(text: str) -> tuple[str, dict[str, int]]:
    text, names = _mask_name(text)
    text, emails = re.subn(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[EMAIL]", text, flags=re.I
    )
    text, phones = re.subn(
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", text
    )
    text, years = re.subn(r"\b(?:19|20)\d{2}\b", "[YEAR]", text)
    text, genders = re.subn(
        r"\b(?:he|she|him|her|his|hers|male|female|man|woman|boy|girl)\b",
        "[GENDER]", text, flags=re.I
    )
    return text, {
        "names_removed": names,
        "emails_removed": emails,
        "phones_removed": phones,
        "years_masked": years,
        "gender_terms_masked": genders,
    }


def run_bias_filter_agent(
    resume: str,
    model_id: str,
    generate_text: Callable[[str, str], str] | None = None,
) -> dict:
    """Always mask first; use at most one bounded model verification call."""
    filtered, stats = bias_filter_rule_based(resume)
    steps = [{
        "action": "mask_pii",
        "source": "deterministic",
        "reason": "Apply the mandatory PII mask once before model evaluation.",
    }]
    issues = []
    tips = []
    if generate_text is None:
        issues = ["Model verification unavailable; only rule-based filtering ran."]
    else:
        prompt = f"""Review this partially de-identified resume for remaining PII
or bias signals. Do not infer protected attributes. Return JSON only with
residual_issues (string array) and fairness_tips (up to 3 strings).

Filtered resume:
\"\"\"{filtered[:4000]}\"\"\""""
        try:
            verified = VerificationResult.model_validate_json(
                clean_json_text(generate_text(prompt, model_id))
            )
            issues = verified.residual_issues
            tips = verified.fairness_tips
            steps.append({
                "action": "verify_and_finish",
                "source": "model",
                "reason": "Check the already-masked resume once for residual risks.",
            })
        except Exception:
            issues = ["Model verification failed; manual privacy review is required."]
            steps.append({
                "action": "verification_failed",
                "source": "model",
                "reason": "The verifier failed or returned invalid JSON; no PII gate was bypassed.",
            })
    return {
        "filtered_resume": filtered,
        "removed_stats": stats,
        "residual_issues": issues,
        "fairness_tips": tips,
        "steps": steps,
    }


def validate_analysis(raw: str, bias_result: dict) -> dict:
    """Reject malformed model output, then attach authoritative filter facts."""
    result = AnalysisResult.model_validate_json(clean_json_text(raw)).model_dump()
    result["bias_filter"] = {
        "removed_stats": bias_result["removed_stats"],
        "residual_issues": bias_result["residual_issues"],
        "fairness_tips": bias_result["fairness_tips"],
        "steps": bias_result["steps"],
    }
    return result


def analysis_schema() -> str:
    return json.dumps(AnalysisResult.model_json_schema(), ensure_ascii=False)
