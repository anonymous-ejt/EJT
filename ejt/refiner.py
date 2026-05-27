from .prompts import PromptBuilder
from .utils import clean_text


class FeedbackRefiner:
    """Improves failed generations using evaluator feedback and few-shot examples."""

    def __init__(self, config, client):
        self.config = config
        self.client = client

    def refine(self, template, query, feedback, fewshot_text) -> str:
        prompt = PromptBuilder.build_refiner_prompt(
            template,
            query,
            feedback,
            fewshot_text,
        )

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return clean_text(response.choices[0].message.content)
