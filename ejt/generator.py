from openai import OpenAI

from .prompts import PromptBuilder
from .utils import clean_text


class EJTGenerator:
    """Creates the initial Embedded Jailbreak Template candidate."""

    def __init__(self, config, client=None):
        self.config = config
        self.client = client or OpenAI()

    def generate(self, template, query) -> str:
        prompt = PromptBuilder.build_generator_prompt(template, query)

        try:
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
        except Exception as error:
            return f"[ERROR] {str(error)}"
