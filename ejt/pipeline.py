from dataclasses import asdict

import pandas as pd
from tqdm.auto import tqdm

from .evaluator import IntentEvaluator, StructureEvaluator
from .generator import EJTGenerator
from .prompts import build_fewshot
from .refiner import FeedbackRefiner
from .schemas import EJTPair
from .utils import clean_text, create_openai_client, ensure_parent_dir


class EJTPipeline:
    """Runs the Generator-Evaluator-Refiner loop for EJT construction."""

    def __init__(self, config, client=None):
        self.config = config
        self.client = client or create_openai_client()

        self.generator = EJTGenerator(config, self.client)
        self.structure_evaluator = StructureEvaluator(config)
        self.intent_evaluator = IntentEvaluator(config, self.client)
        self.refiner = FeedbackRefiner(config, self.client)

        self.fewshot_pool = []

    def initialize_pairs(self, template_df, query_df):
        pairs = []

        for t_idx, t_row in template_df.iterrows():
            template = clean_text(t_row[self.config.template_column])

            for q_idx, q_row in query_df.iterrows():
                query = clean_text(q_row[self.config.query_column])
                pair = EJTPair(
                    pair_id=f"T{t_idx}_Q{q_idx}",
                    template=template,
                    query=query,
                    category=clean_text(q_row.get("Category", "")),
                    subcategory=clean_text(q_row.get("Subcategory", "")),
                )

                pairs.append(pair)

        return pairs

    def initial_generation(self, pairs) -> None:
        for pair in tqdm(pairs, desc="Generator"):
            pair.generated_template = self.generator.generate(
                pair.template,
                pair.query,
            )

            if pair.generated_template.startswith("[ERROR]"):
                pair.status = "error"

    def apply_refinement(self, pair, feedback, fewshot_text) -> None:
        pair.generated_template = self.refiner.refine(
            template=pair.template,
            query=pair.query,
            feedback=feedback,
            fewshot_text=fewshot_text,
        )

    def run_round(self, pairs, round_id) -> None:
        fewshot_text = build_fewshot(
            self.fewshot_pool,
            self.config.fewshot_top_k,
        )

        for pair in tqdm(pairs, desc=f"Evaluator/Refiner Round {round_id}"):
            if pair.status in {"pass", "error"}:
                continue

            structure_result = self.structure_evaluator.evaluate(
                pair.template,
                pair.generated_template,
            )

            if structure_result["decision"] == "fail":
                self.apply_refinement(
                    pair,
                    structure_result["feedback"],
                    fewshot_text,
                )
                continue

            intent_result = self.intent_evaluator.evaluate(
                pair.query,
                pair.generated_template,
            )

            if intent_result["decision"] == "fail":
                self.apply_refinement(
                    pair,
                    intent_result["feedback"],
                    fewshot_text,
                )
                continue

            pair.status = "pass"
            self.fewshot_pool.append(pair)

    def save_final(self, pairs) -> None:
        ensure_parent_dir(self.config.final_output_path)

        df = pd.DataFrame([asdict(pair) for pair in pairs])
        df["t_idx"] = df["pair_id"].apply(lambda x: int(x.split("_")[0][1:]))
        df["q_idx"] = df["pair_id"].apply(lambda x: int(x.split("_")[1][1:]))
        df = df.sort_values(["t_idx", "q_idx"]).drop(columns=["t_idx", "q_idx"])

        df.to_csv(
            self.config.final_output_path,
            index=False,
            encoding="utf-8-sig",
        )

    def run(self, template_df=None, query_df=None) -> None:
        if template_df is None:
            template_df = pd.read_csv(
                self.config.template_path,
                encoding="utf-8-sig",
            )

        if query_df is None:
            query_df = pd.read_csv(
                self.config.query_path,
                encoding="utf-8-sig",
            )

        pairs = self.initialize_pairs(template_df, query_df)
        self.initial_generation(pairs)

        for round_id in range(self.config.max_rounds):
            self.run_round(pairs, round_id)

            if all(pair.status == "pass" for pair in pairs):
                print("All pairs passed.")
                self.save_final(pairs)
                return

        for pair in pairs:
            if pair.status != "pass":
                pair.status = "fail"

        print("Reached maximum refinement rounds.")
        self.save_final(pairs)
