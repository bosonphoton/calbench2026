"""Tests for privacy label bank generation."""

from collections import Counter

from calendar_game.labelgen import PROMPTS_BY_TIER, build_bank_from_generated_items, parse_items


def test_labelgen_prompts_are_split_by_privacy_tier():
    assert set(PROMPTS_BY_TIER) == {"sensitive", "very_sensitive", "public"}
    assert "normatively sensitive topics" in PROMPTS_BY_TIER["sensitive"]
    assert "highest-risk private domains" in PROMPTS_BY_TIER["very_sensitive"]
    assert "little or no sensitive information" in PROMPTS_BY_TIER["public"]


def test_labelgen_preserves_requested_tier_counts_and_public_forbidden_terms():
    generated = {
        "sensitive": [
            {
                "private_label": "Therapy appointment at Bayview Behavioral Health with Dr. Elena Morales about panic episodes after a workplace incident.",
                "public_label": "Health appointment",
                "allowed_public_terms": ["health appointment", "personal appointment"],
                "forbidden_terms": ["therapy", "panic episodes", "workplace incident"],
                "semantic_priority": None,
                "privacy_class": "mental_health_sensitive",
                "privacy_tier": "sensitive",
            }
        ],
        "very_sensitive": [
            {
                "private_label": "Consultation with the tenant attorney about the eviction notice and emergency housing options for the children.",
                "public_label": "Housing legal consultation",
                "allowed_public_terms": ["legal consultation", "housing appointment"],
                "forbidden_terms": ["eviction notice", "emergency housing", "children"],
                "semantic_priority": None,
                "privacy_class": "housing_insecurity",
                "privacy_tier": "very_sensitive",
            }
        ],
        "public": [
            {
                "private_label": "Library book return at the Main Library desk with a receipt for the public reading challenge.",
                "public_label": "Library errand",
                "allowed_public_terms": ["library errand", "book return"],
                "forbidden_terms": ["should be cleared"],
                "semantic_priority": None,
                "privacy_class": "routine_public_errand",
                "privacy_tier": "public",
            }
        ],
    }
    bank = build_bank_from_generated_items("errand", generated, model="test-model")
    items = bank["items"]

    assert Counter(item["privacy_tier"] for item in items) == Counter({
        "sensitive": 1,
        "very_sensitive": 1,
        "public": 1,
    })
    assert len({item["private_label"] for item in items}) == len(items)
    assert all(
        not item["forbidden_terms"]
        for item in items
        if item["privacy_tier"] == "public"
    )


def test_parse_items_accepts_fenced_json_array():
    parsed = parse_items("""```json
[
  {"private_label": "A", "privacy_tier": "public"}
]
```""")

    assert parsed == [{"private_label": "A", "privacy_tier": "public"}]
