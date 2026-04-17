# llmeval

An open-source Python framework for behavioral regression testing of LLM-powered applications.

Define test suites in YAML, run them against any OpenAI or Anthropic model, score outputs with an LLM-as-judge, and catch prompt regressions in CI before they reach production.

---

## Quickstart

```bash
pip install llmeval  # coming soon to PyPI

export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

llmeval run --suite tests/fixtures/example_suite.yaml
```

---

## Test Suite Format

```yaml
suite:
  name: "Customer Support Bot - Tone Tests"
  version: "1.0.0"
  model: "claude-sonnet-4-20250514"
  judge_model: "claude-sonnet-4-20250514"

tests:
  - id: "tone-empathy-001"
    description: "Response should be empathetic when user is frustrated"
    prompt: "I've been waiting 3 weeks for my order and nobody will help me!"
    rubric:
      criteria:
        - name: "empathy"
          description: "Response acknowledges the user's frustration"
          weight: 0.4
        - name: "actionability"
          description: "Response offers a concrete next step"
          weight: 0.4
        - name: "tone"
          description: "Response is professional and not defensive"
          weight: 0.2
      passing_threshold: 0.75
    tags: ["tone", "empathy", "regression-critical"]
```

---

## Features

- **YAML/JSON test definitions** — version-controlled, human-readable test suites
- **Multi-provider support** — OpenAI and Anthropic out of the box
- **LLM-as-judge scoring** — rubric-based evaluation with configurable weights
- **Regression detection** — compare scores across model versions or prompt changes
- **CI/CD integration** — GitHub Actions workflow included
- **Rich CLI output** — pass/fail/regression summary in your terminal
- **React dashboard** — visual report browser (coming soon)

---

## Environment Variables

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
LLMEVAL_DB_PATH=./llmeval.db        # defaults to local SQLite
LLMEVAL_JUDGE_MODEL=claude-sonnet-4-20250514
RUN_INTEGRATION_TESTS=0
```

---

## Development

```bash
git clone https://github.com/emilysuh/llm-eval-framework
cd llm-eval-framework
poetry install
poetry run pytest
```

---

## License

MIT
