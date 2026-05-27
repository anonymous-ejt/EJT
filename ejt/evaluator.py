from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .prompts import PromptBuilder
from .utils import clean_text, safe_json_loads


class SimilarityMetrics:
    @staticmethod
    def tfidf_similarity(text1, text2) -> float:
        text1 = clean_text(text1)
        text2 = clean_text(text2)

        if not text1 and not text2:
            return 1.0

        if not text1 or not text2:
            return 0.0

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([text1, text2])

        return float(cosine_similarity(vectors)[0, 1])

    @staticmethod
    def levenshtein_similarity(text1, text2) -> float:
        text1 = clean_text(text1)
        text2 = clean_text(text2)

        distance = Levenshtein.distance(text1, text2)
        denominator = max(len(text1), len(text2)) or 1

        return 1.0 - (distance / denominator)


class StructureEvaluator:
    """Checks whether the generated template preserves the source structure."""

    def __init__(self, config):
        self.config = config

    def evaluate(self, original_template, generated_template):
        tfidf_score = SimilarityMetrics.tfidf_similarity(
            original_template,
            generated_template,
        )
        ld_score = SimilarityMetrics.levenshtein_similarity(
            original_template,
            generated_template,
        )

        passed = (
            tfidf_score >= self.config.tfidf_threshold
            and ld_score >= self.config.levenshtein_threshold
        )

        if passed:
            return {
                "decision": "pass",
                "reason": "structure preserved",
                "feedback": "Structure similarity sufficient.",
                "tfidf": tfidf_score,
                "ld": ld_score,
            }

        return {
            "decision": "fail",
            "reason": "structure diverged",
            "feedback": (
                "Keep wording and sentence layout closer to the original "
                "template."
            ),
            "tfidf": tfidf_score,
            "ld": ld_score,
        }


class IntentEvaluator:
    """Checks whether the generated template reflects the query intent."""

    def __init__(self, config, client):
        self.config = config
        self.client = client

    def evaluate(self, query, generated_template):
        prompt = PromptBuilder.build_intent_prompt(query, generated_template)

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        parsed = safe_json_loads(response.choices[0].message.content)

        if parsed is None:
            return {
                "decision": "fail",
                "reason": "invalid_json",
                "feedback": "Evaluator output was not valid JSON.",
            }

        decision = clean_text(parsed.get("decision", "fail")).lower()

        if decision not in {"pass", "fail"}:
            decision = "fail"

        return {
            "decision": decision,
            "reason": clean_text(parsed.get("reason", "")),
            "feedback": clean_text(
                parsed.get(
                    "feedback",
                    "Reflect the topic intent more clearly.",
                )
            ),
        }
