# Embedded Jailbreak Templates

This repository implements a Generator-Evaluator-Refiner pipeline for
constructing Embedded Jailbreak Templates (EJT).

## Architecture

```text
Template + Query
      |
      v
Generator
      |
      v
Evaluator
  - structure preservation
  - intent alignment
      |
      v
Refiner
  - feedback-driven rewrite
  - few-shot examples from passed samples
      |
      v
Final EJT dataset
```

## Components

- `ejt/generator.py`: creates the initial EJT candidate from a template-query
  pair.
- `ejt/evaluator.py`: checks whether the generated template preserves structure
  and reflects the query intent.
- `ejt/refiner.py`: rewrites failed candidates using evaluator feedback.
- `ejt/pipeline.py`: orchestrates generation, evaluation, iterative refinement,
  and CSV export.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your OpenAI API key in the environment, then fill in the paths and column
names in `run.py`:

```python
config = Config(
    model_name="gpt-...",
    template_path="data/templates.csv",
    query_path="data/queries.csv",
    template_column="Template",
    query_column="Query",
    output_dir="outputs",
    output_filename="ejt_results.csv",
)
```

Run the pipeline:

```bash
python run.py
```
