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

Set your API key in the environment:

```bash
export GPT_API_KEY="your_api_key_here"
```

On Windows PowerShell:

```powershell
$env:GPT_API_KEY="your_api_key_here"
```

The repository includes two example CSV files:

- `data/Example_Template_for_EJT.csv`
- `data/Example_Query_for_EJT.csv`

`run.py` is already configured to use these files with the `prompt` and `Query`
columns, so you can run the example directly:

```bash
python run.py
```

By default, outputs are written to:

```text
outputs/ejt_results.csv
```

You can override the model without editing the code:

```bash
export OPENAI_MODEL="gpt-5.4"
```
