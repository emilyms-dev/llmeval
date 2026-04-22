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

## CLI Reference

### `llmeval run`

Run a test suite and score outputs with LLM-as-judge.

```bash
llmeval run --suite suite.yaml
llmeval run --suite suite.yaml --model gpt-4o
llmeval run --suite suite.yaml --tag regression-critical
llmeval run --suite suite.yaml --label commit=abc123 --label branch=main --label pr=42
llmeval run --suite suite.yaml --no-save          # skip persisting to DB
llmeval run --suite suite.yaml --concurrency 10
llmeval run --suite suite.yaml --samples 3        # call judge 3×, report median score + stddev
llmeval run --suite suite.yaml --samples 3 --temperature 0.7  # non-zero temp for variance
```

`--samples` calls the judge model N times per test and reports the **median** score with a per-criterion standard deviation. Use `--temperature` (default `0.0`) to introduce sampling variance when `--samples > 1`.

Exit codes: `0` all pass · `1` any failure or error · `2` config/IO error

### `llmeval list`

Browse stored runs.

```bash
llmeval list
llmeval list --status failed
llmeval list --suite "Customer Support Bot - Tone Tests"
llmeval list --model gpt-4o
llmeval list --tag regression-critical
llmeval list --limit 25
llmeval list --format json | jq '.[0].run_id'   # machine-readable for CI
```

### `llmeval show`

Display a stored run in full.

```bash
llmeval show <run-id>
llmeval show abc123ef          # prefix works if unambiguous
```

### `llmeval diff`

Side-by-side comparison of two runs.

```bash
llmeval diff <run-id-a> <run-id-b>
llmeval diff <run-id>           # omit B to auto-compare against previous run
```

### `llmeval compare`

CI-oriented regression check with machine-readable output.

```bash
llmeval compare <run-id-a> <run-id-b>
llmeval compare <run-id-a> <run-id-b> --fail-on-regression   # exit 1 if regressions
llmeval compare <run-id-a> <run-id-b> --json                 # machine-readable output
llmeval compare <run-id-a> <run-id-b> --fail-on-regression --json
```

JSON output schema:

```json
{
  "run_a": {"run_id": "...", "suite_name": "...", "model": "...", "total_tests": 10, "passed_tests": 9},
  "run_b": {"run_id": "...", "suite_name": "...", "model": "...", "total_tests": 10, "passed_tests": 8},
  "total": 10,
  "regressions": 1,
  "improvements": 0,
  "unchanged": 9,
  "tests": [
    {"test_id": "t1", "is_regression": true, "is_improvement": false, "score_delta": -0.3, "score_a": 0.9, "score_b": 0.6}
  ]
}
```

### `llmeval latest`

Show the most recent completed run.

```bash
llmeval latest
llmeval latest --suite "My Suite"
```

### `llmeval rerun`

Re-run a suite using the exact configuration of a previous run.

```bash
llmeval rerun <run-id>
llmeval rerun <run-id> --label commit=def456   # overlay new labels
llmeval rerun <run-id> --concurrency 10
llmeval rerun <run-id> --samples 3 --temperature 0.5
```

### `llmeval cancel`

Cancel a pending or running job started via `llmeval serve`.

```bash
llmeval cancel <run-id>
```

### `llmeval export`

Export a run to JSON or CSV.

```bash
llmeval export <run-id>                        # JSON to stdout
llmeval export <run-id> --format csv
llmeval export <run-id> --format json --out run.json
llmeval export <run-id> --format csv | grep FAIL
```

### `llmeval prune`

Remove old runs from storage.

```bash
llmeval prune --older-than 30d            # dry run — shows what would be deleted
llmeval prune --older-than 7d --status failed --yes   # actually delete
```

### `llmeval init`

Scaffold a new project in the current directory.

```bash
llmeval init          # creates tests/fixtures/example_suite.yaml and .env.example
llmeval init --force  # overwrite existing files
```

### Shell Completion

Enable tab completion for your shell:

```bash
llmeval --install-completion   # install for the detected shell (bash/zsh/fish)
llmeval --show-completion      # print the completion script
```

### `llmeval serve`

Start the dashboard API server.

```bash
llmeval serve
llmeval serve --host 0.0.0.0 --port 9000
llmeval serve --db /path/to/llmeval.db
llmeval serve --reload    # development auto-reload
```

---

## Dashboard API

Start the server with `llmeval serve`, then open the React dashboard at `http://localhost:5173` (after `npm run dev` in `dashboard/`).

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/suites` | Discover local YAML suite files |
| `GET` | `/api/runs` | Paginated, filtered run list |
| `POST` | `/api/runs` | Trigger a new eval run (async) |
| `GET` | `/api/runs/{run_id}/status` | Lightweight status polling |
| `POST` | `/api/runs/{run_id}/cancel` | Cancel a pending or running job |
| `GET` | `/api/runs/{run_id}` | Full run with all results |
| `GET` | `/api/runs/{run_id}/previous` | Previous run of same suite |
| `GET` | `/api/runs/{run_id}/export` | Download as JSON or CSV |
| `GET` | `/api/runs/{run_id_a}/diff/{run_id_b}` | Per-test diff between two runs |

### Trigger a run (POST /api/runs)

```json
{
  "suite_path": "tests/fixtures/example_suite.yaml",
  "model": "gpt-4o",
  "tags": ["regression-critical"],
  "labels": {"commit": "abc123", "branch": "main", "pr": "42"},
  "concurrency": 5,
  "timeout": 1800
}
```

Returns `202 Accepted` with `{"run_id": "...", "status": "pending"}`. Poll `GET /api/runs/{run_id}/status` until `status` is `completed`, `failed`, or `cancelled`.

Returns `409 Conflict` if a run for the same suite is already pending or running.

### Filter runs (GET /api/runs)

```
GET /api/runs?suite=My+Suite&model=gpt-4o&status=completed&tag=ci&date_from=2024-01-01&limit=50
```

---

## Run Lifecycle

```
pending → running → completed
                 ↘ failed
                 ↘ cancelled
```

- **pending** — saved to storage, pipeline not yet started
- **running** — pipeline actively executing
- **completed** — all tests ran and were scored (individual tests may still have failed)
- **failed** — pipeline error (suite load failure, adapter error, timeout)
- **cancelled** — stopped by user via `llmeval cancel` or `POST /api/runs/{run_id}/cancel`

Runs time out after 30 minutes by default (configurable via `timeout` in the API request).

---

## CI Integration

### GitHub Actions

```yaml
# .github/workflows/eval.yml
- name: Run eval suite
  run: |
    llmeval run --suite tests/fixtures/example_suite.yaml \
      --label commit=${{ github.sha }} \
      --label branch=${{ github.ref_name }} \
      --label pr=${{ github.event.pull_request.number }}

- name: Compare against baseline
  run: |
    BASELINE=$(llmeval list --status completed --limit 1 | awk 'NR==2{print $1}')
    CANDIDATE=$(llmeval list --status completed --limit 1 --suite "My Suite" | awk 'NR==2{print $1}')
    llmeval compare $BASELINE $CANDIDATE --fail-on-regression --json > regression-report.json
```

### Run Labels

Attach arbitrary key-value metadata to any run for traceability:

```bash
llmeval run --suite suite.yaml \
  --label commit=abc123 \
  --label branch=feature/new-prompt \
  --label pr=99 \
  --label author=emily
```

Labels are stored with the run and visible in `llmeval show`, the dashboard, and exports.

---

## Features

- **YAML/JSON test definitions** — version-controlled, human-readable test suites
- **Multi-provider support** — OpenAI and Anthropic out of the box
- **LLM-as-judge scoring** — rubric-based evaluation with configurable weights and thresholds
- **Regression detection** — compare scores across model versions, prompt changes, or runs
- **Run labels** — attach CI metadata (commit, branch, PR) to every run
- **Duplicate-run protection** — API rejects new jobs when the same suite is already active
- **Cancellation** — cancel pending or running jobs via CLI or API
- **Configurable timeouts** — runs fail cleanly after a configurable deadline (default 30 min)
- **Filtering & export** — filter runs by suite, model, status, tag, or date; export to JSON or CSV
- **CI/CD integration** — `compare --fail-on-regression --json` for pipeline-friendly output
- **GitHub Actions workflow** — `eval.yml` included
- **Rich CLI output** — pass/fail/regression summary in your terminal
- **React dashboard** — visual report browser with filtering, diff view, and export

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

## Security

`llmeval serve` is designed for local and trusted-network use.

**Authentication.** Set `LLMEVAL_API_TOKEN` in your environment to enable bearer-token
auth on all mutating endpoints (`POST /api/runs`, `POST /api/runs/{id}/cancel`). When
unset, the server runs in **open mode** and logs a startup warning — acceptable for local
development, never for production.

```bash
export LLMEVAL_API_TOKEN=$(openssl rand -hex 32)
llmeval serve
# Dashboard: set VITE_LLMEVAL_API_TOKEN in dashboard/.env.local
```

**CORS.** The default `Access-Control-Allow-Origin` is `http://localhost:5173` (the Vite
dev server). Override with `LLMEVAL_CORS_ORIGIN` (comma-separated for multiple origins).

**Path traversal.** `POST /api/runs` resolves `suite_path` relative to `LLMEVAL_SUITES_DIR`
(default: CWD). Any path that escapes that directory is rejected with 422 before
reaching the filesystem.

**Rate limiting.** `POST /api/runs` is limited to 10 requests per minute per IP.

**Production checklist:**
- Set `LLMEVAL_API_TOKEN` to a strong random value.
- Terminate TLS at a reverse proxy (nginx, Caddy) — never expose uvicorn directly.
- Restrict `LLMEVAL_CORS_ORIGIN` to your dashboard's actual origin.
- Set `LLMEVAL_SUITES_DIR` to the directory containing your suite files.

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
