import os
from dataclasses import dataclass


@dataclass
class Config:
    """Runtime configuration for the EJT pipeline."""

    model_name: str = ""

    template_path: str = ""
    query_path: str = ""

    template_column: str = ""
    query_column: str = ""

    output_dir: str = ""
    output_filename: str = ""

    max_rounds: int = 10

    tfidf_threshold: float = 0.5
    levenshtein_threshold: float = 0.4

    fewshot_top_k: int = 3

    @property
    def final_output_path(self) -> str:
        return os.path.join(self.output_dir, self.output_filename)
