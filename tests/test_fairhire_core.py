import json
import unittest

from pydantic import ValidationError

from fairhire_core import (
    bias_filter_rule_based,
    run_bias_filter_agent,
    validate_analysis,
)


RESUME = (
    "Alex Taylor\n"
    "Email: alex.taylor@example.com | Phone: 555-123-4567\n"
    "Software engineer with 6 years of Python and SQL experience.\n"
)


def valid_analysis():
    return {
        "summary_bullets": ["Python and SQL experience"],
        "match_score": 85,
        "skills_fit": {
            "strong": ["Python", "SQL"],
            "medium": [],
            "missing": ["stakeholder communication"],
        },
        "bias_flags": [],
        "recommendation": "Review the work samples; do not use this as a hiring decision.",
    }


class MaskingTests(unittest.TestCase):
    def test_masks_header_name_email_and_phone(self):
        filtered, stats = bias_filter_rule_based(RESUME)
        self.assertIn("[NAME]", filtered)
        self.assertIn("[EMAIL]", filtered)
        self.assertIn("[PHONE]", filtered)
        self.assertNotIn("Alex Taylor", filtered)
        self.assertNotIn("alex.taylor@example.com", filtered)
        self.assertEqual(stats["names_removed"], 1)
        self.assertEqual(stats["emails_removed"], 1)
        self.assertEqual(stats["phones_removed"], 1)

    def test_masks_explicit_name_label(self):
        filtered, stats = bias_filter_rule_based("Name: Jane Doe\nPython developer")
        self.assertIn("Name: [NAME]", filtered)
        self.assertNotIn("Jane Doe", filtered)
        self.assertEqual(stats["names_removed"], 1)

    def test_does_not_treat_job_title_as_name(self):
        filtered, stats = bias_filter_rule_based(
            "Software Engineer\nEmail: candidate@example.com\n"
        )
        self.assertTrue(filtered.startswith("Software Engineer"))
        self.assertEqual(stats["names_removed"], 0)

    def test_keeps_unrecognized_name_layout_for_review(self):
        filtered, stats = bias_filter_rule_based("Dr. A. Taylor\nPython experience")
        self.assertIn("Dr. A. Taylor", filtered)
        self.assertEqual(stats["names_removed"], 0)


class AgentTests(unittest.TestCase):
    def test_one_verification_call_sees_only_masked_resume(self):
        calls = []

        def fake_model(prompt, model_id):
            calls.append((prompt, model_id))
            return json.dumps({"residual_issues": [], "fairness_tips": []})

        result = run_bias_filter_agent(RESUME, "test-model", fake_model)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("Alex Taylor", calls[0][0])
        self.assertNotIn("alex.taylor@example.com", calls[0][0])
        self.assertEqual([step["action"] for step in result["steps"]], [
            "mask_pii", "verify_and_finish"
        ])

    def test_verifier_failure_preserves_mask_and_flags_review(self):
        def broken_model(prompt, model_id):
            raise TimeoutError("provider timeout")

        result = run_bias_filter_agent(RESUME, "test-model", broken_model)
        self.assertNotIn("Alex Taylor", result["filtered_resume"])
        self.assertIn("manual privacy review", result["residual_issues"][0])
        self.assertEqual(result["steps"][-1]["action"], "verification_failed")


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.bias = run_bias_filter_agent(RESUME, "test-model")

    def test_accepts_valid_analysis_and_attaches_authoritative_stats(self):
        result = validate_analysis(json.dumps(valid_analysis()), self.bias)
        self.assertEqual(result["match_score"], 85)
        self.assertEqual(result["bias_filter"]["removed_stats"]["names_removed"], 1)

    def test_rejects_out_of_range_score(self):
        payload = valid_analysis()
        payload["match_score"] = 101
        with self.assertRaises(ValidationError):
            validate_analysis(json.dumps(payload), self.bias)

    def test_rejects_missing_fields(self):
        payload = valid_analysis()
        del payload["skills_fit"]
        with self.assertRaises(ValidationError):
            validate_analysis(json.dumps(payload), self.bias)

    def test_rejects_extra_model_control_field(self):
        payload = valid_analysis()
        payload["bias_filter"] = {"removed_stats": {"names_removed": 0}}
        with self.assertRaises(ValidationError):
            validate_analysis(json.dumps(payload), self.bias)

    def test_rejects_score_string(self):
        payload = valid_analysis()
        payload["match_score"] = "85"
        with self.assertRaises(ValidationError):
            validate_analysis(json.dumps(payload), self.bias)


if __name__ == "__main__":
    unittest.main()
