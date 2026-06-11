import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "FDE-Assignment"))

from research_mini_lite.evaluation import QUALITY_WEIGHTS
from research_mini_lite.evaluation import _normalize_quality_scores
from research_mini_lite.evaluation import _tavily_research_output_schema


class EvaluationHelpersTest(unittest.TestCase):
    def test_quality_normalization_uses_documented_weights(self):
        self.assertEqual(
            QUALITY_WEIGHTS,
            {
                "completeness": 0.25,
                "grounding": 0.20,
                "source_quality": 0.05,
                "synthesis": 0.10,
                "clarity": 0.15,
                "latency": 0.25,
            },
        )

        judged = {
            "scores": {
                "research_mini_lite": {
                    "overall": 1,
                    "completeness": 4,
                    "grounding": 3,
                    "source_quality": 5,
                    "synthesis": 4,
                    "clarity": 4,
                    "latency": 5,
                },
                "tavily_research_mini": {
                    "overall": 5,
                    "completeness": 5,
                    "grounding": 5,
                    "source_quality": 5,
                    "synthesis": 5,
                    "clarity": 5,
                    "latency": 2,
                },
            },
            "winner": "tavily_research_mini",
        }

        normalized = _normalize_quality_scores(judged)

        self.assertEqual(normalized["scores"]["research_mini_lite"]["overall"], 4.1)
        self.assertEqual(normalized["scores"]["tavily_research_mini"]["overall"], 4.2)
        self.assertEqual(normalized["winner"], "tavily_research_mini")
        self.assertEqual(normalized["scoring"]["weights"], QUALITY_WEIGHTS)

    def test_tavily_schema_normalization_adds_required_descriptions(self):
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string", "description": "Source URL."},
                        },
                    },
                },
            },
            "required": ["summary", "citations", "ignored_missing_field"],
        }

        normalized = _tavily_research_output_schema(schema)

        self.assertEqual(normalized["required"], ["summary", "citations"])
        self.assertEqual(normalized["properties"]["summary"]["description"], "The summary field.")
        citation_item = normalized["properties"]["citations"]["items"]
        self.assertEqual(citation_item["description"], "An item in citations.")
        self.assertEqual(citation_item["properties"]["title"]["description"], "The title field.")
        self.assertEqual(citation_item["properties"]["url"]["description"], "Source URL.")


if __name__ == "__main__":
    unittest.main()
