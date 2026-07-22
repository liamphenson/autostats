# AutoStats

AutoStats is an AI agent that runs rigorous statistical analyses on real, user-provided
datasets. It never computes statistics itself or invents numbers: every statistic,
p-value, model coefficient, or forecast it reports comes from an actual tool call, and
the LLM's job is to choose the right tool, chain calls together (upload → describe →
test → report), and narrate results it is only allowed to *quote or closely paraphrase*
from each tool's deterministic `interpretation` field.

See `src/autostats/core/agent/prompts.py` for the exact rules the model operates under.

## How it's put together

| Piece | File | Role |
|---|---|---|
| Agent loop | `core/agent/loop.py` (`AutoStatsAgent.run_turn`) | Runs the OpenAI **Responses API** tool-calling loop for one conversational turn (bounded by `MAX_TOOL_ITERATIONS = 8`); yields typed events (`TextDelta`, `ToolCallStarted`, `ToolResultReady`, `PlotReady`, `TurnComplete`, `ErrorEvent`). |
| Tool registry | `core/tools/registry.py`, `core/tools/base.py` | Every tool is a `BaseTool` subclass with a pydantic `input_model`; `@REGISTRY.register` adds it, `ToolRegistry.dispatch()` validates arguments and runs it. No separate schema-alignment step is needed — pydantic *is* the schema. |
| Data manager | `core/data/manager.py` | Owns every loaded `DataFrame` for a session. Raw data never reaches the LLM — only `DatasetMeta` (shape, columns, a small preview) does, so the model must reference a `dataset_id` instead of inventing data. |
| Session store | `core/session/store.py` | SQLite-backed message history + accumulated `AnalysisResult`s, keyed by `session_id`. |
| Report builder | `core/reporting/builder.py` | Renders a session's accumulated results to HTML, PDF, or a Jupyter notebook. |

## Submission info

- **Build path:** a custom handler wrapping existing code (not a hand-authored tool loop).
- **Entry-point file / class:** `handler.py`, class `AutoStatsHandler`.
- **Input / output:** takes the user's text request plus any attached dataset file(s)
  (CSV/TSV/Excel/JSON/Parquet); returns a dict with `answer` (human-readable text),
  `tools_used` (the tool-call trace), and optionally `plots` (base64-encoded PNGs).
- **Verified locally:** runs and returns a completed result with a populated `answer`.
- **Dependencies:** Python packages via `pyproject.toml` (see Install below). One
  system-binary dependency: the optional `report-pdf` extra (`weasyprint`) needs system
  libraries (Pango, cairo, gdk-pixbuf) — only needed for PDF report export; HTML/notebook
  export don't need it.
- **Hardware:** none — no GPU or unusual memory requirements.
- **Secrets:** only `OPENAI_API_KEY` (see Install below); no keys or `.env` files are
  included in this repo.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                    # core deps only
# pip install -e ".[timeseries,report-pdf,notebook,web,effect-size,data-sources,dev]"
```

Create a `.env` file at the repo root with your OpenAI key:

```
OPENAI_API_KEY=sk-...
```

`Settings.openai_api_key` (`core/config.py`) is aliased directly to `OPENAI_API_KEY` —
note that this **bypasses** the `AUTOSTATS_` prefix every other setting uses. Other
useful overrides, same `.env` file:

```
AUTOSTATS_OPENAI_MODEL=gpt-4o-mini   # default is gpt-5 — override for cheaper/faster runs
AUTOSTATS_DATA_DIR=./data            # where sessions.db, datasets, and plots are written (relative to cwd)
```

## Quickstart (CLI)

```bash
autostats chat                                  # start an interactive session
autostats upload path/to/data.csv               # load a dataset, then chat about it
autostats datasets <session_id>                 # list datasets loaded in a session
autostats report <session_id> --output out.html --fmt html   # export accumulated results
```

`chat` and `upload` share one `AutoStatsAgent` per process (`cli/main.py:_agent`), backed
by `sessions.db` under `AUTOSTATS_DATA_DIR`/`sessions`. Resume a session with
`autostats chat --session-id <id>`.

## Capabilities (current tool catalog)

Every tool below is registered via `core/tools/__init__.py:load_all_tools()`. Adding a
new one is: write a `BaseTool` subclass + pydantic `input_model`, decorate with
`@REGISTRY.register`, add one import line to `load_all_tools()` — no agent-loop or
prompt changes required.

**Data I/O** (`data_io`)
- `upload_dataset` — load CSV/TSV/Excel/JSON/Parquet into the session's dataset store
- `list_datasets` — list datasets loaded in this session
- `get_dataset_preview` — first N rows + dtypes of a loaded dataset

**Descriptive** (`descriptive`)
- `describe_dataset` — count/mean/std/min/max/quartiles for numeric columns
- `correlation_matrix` — Pearson or Spearman correlation matrix

**Parametric hypothesis tests** (`hypothesis_test`)
- `one_sample_t_test`, `two_sample_t_test` (Welch by default), `paired_t_test`
- `one_way_anova` + `pairwise_tukey_posthoc` (post-hoc after a significant ANOVA)
- `pearson_correlation_test`

**Nonparametric alternatives** (`hypothesis_test`)
- `mann_whitney_u_test` (↔ two-sample t), `wilcoxon_signed_rank_test` (↔ paired t),
  `kruskal_wallis_test` (↔ one-way ANOVA), `spearman_correlation_test` (↔ Pearson)
- `chi_square_test_independence`, `chi_square_goodness_of_fit`

**Regression** (`regression`)
- `linear_regression` — OLS with VIF, Durbin-Watson, Breusch-Pagan diagnostics
- `logistic_regression` — binary outcomes, reports odds ratios + McFadden pseudo-R²

**Time series** (`timeseries`)
- `check_stationarity` — ADF + KPSS (their disagreement is itself surfaced as informative)
- `decompose_time_series` — STL trend/seasonal/residual, with a plot
- `fit_arima_model` (auto-order via `pmdarima` if installed, else `(1,1,1)`) → `forecast_time_series`

**Every hypothesis-test/regression/time-series tool** returns (`core/schemas/stat_result.py`):
`assumptions` (a list of checked, not assumed, conditions — normality, variance
homogeneity, sample size, multicollinearity, independence-by-design, etc.),
`assumptions_met`, and `recommended_alternative` when they're violated. The system
prompt requires the model to surface a failed assumption and its recommended
alternative rather than silently reporting the primary test.

### Declared but not yet implemented

These have dependencies listed in `pyproject.toml` and/or fields already in the
schemas, but no working code behind them yet — useful to know before assuming a
capability exists:

- **External data sources** — `DatasetMeta.source` already accepts `"fred"`,
  `"world_bank"`, `"census"`, `"web_scrape"`, and the `data-sources` extra
  (`fredapi`, `wbdata`, `requests`, `beautifulsoup4`, `lxml`) is declared, but
  `core/tools/data/sources/` is empty — only `upload_dataset` (source=`"upload"`)
  actually works today.
- **Web/API surface** — the `web` extra (`fastapi`, `uvicorn`, `sse-starlette`) is
  declared and `core/api/routers/` exists, but contains no route files yet.
- **`effect-size` extra** (`pingouin`) — declared but unused; all current effect
  sizes (Cohen's d, eta², Cramér's V, McFadden pseudo-R²) are computed directly with
  `scipy`/`statsmodels`.
- **`autostats.notebook` package** — no files yet. (Note: this is different from
  report export's `fmt="notebook"`, which *does* work today via `nbformat` directly
  in `ReportBuilder._build_notebook`.)

## Testing

```bash
pytest                          # fast, offline, no API key needed
```

| File | Covers |
|---|---|
| `test_registry.py` | Tool registration mechanics |
| `test_data_manager_and_upload.py` | `DataManager` + `upload_dataset` |
| `test_assumptions.py` | Shared assumption-check primitives |
| `test_parametric.py`, `test_nonparametric.py`, `test_regression.py`, `test_timeseries.py` | Each tool's statistics against known inputs |
| `test_report_builder.py` | HTML/notebook rendering |
| `test_end_to_end_session.py` | Full upload → analyze → report flow, **calling tools directly** (bypasses the LLM) |

**None of these exercise the actual LLM tool-calling loop** — by design, so the suite
stays fast and free to run. To verify the *agent* (not just the tools) end to end,
against a real model, use `autostats chat`: attach/reference a dataset and ask a
question, then watch that it calls `upload_dataset` → `describe_dataset` → a test tool
in sequence, and that its prose matches the tool's `interpretation` field verbatim or
near-verbatim (that's the grounding guarantee — if it doesn't, something's wrong with
the prompt or the loop).

## Debugging tips

- **Tool errors don't crash a turn.** `run_turn` catches any exception raised inside
  `REGISTRY.dispatch` and feeds `{"error": str(exc)}` back to the model as the tool
  result, so the model can react (e.g. re-check a column name) instead of the whole
  request failing. A *turn-level* exception (e.g. the OpenAI call itself failing)
  does end the turn early with an `ErrorEvent` and no `TurnComplete`.
- **Where state lives:** `AUTOSTATS_DATA_DIR/sessions.db` (SQLite — messages +
  accumulated `AnalysisResult`s), `AUTOSTATS_DATA_DIR/sessions/<id>/datasets/*.parquet`
  (registered dataframes), `AUTOSTATS_DATA_DIR/sessions/<id>/plots/*.png`. Deleting a
  session's row from `sessions.db` does **not** delete its parquet/plot files.
  `DataManager` only rebuilds its in-memory dataset *catalog* (`_meta`, used for the
  system prompt's `catalog_text()`) from `register()` calls in the current process —
  it does not re-scan disk on restart, so a fresh process restarts with catalogs empty
  even though the parquet files are still there.
  `two_sample_t_test`/`mann_whitney_u_test`/etc. require **exactly 2** groups in
  `group_col`, or the tool raises (surfaced to the model as a tool error, per above).
- **ARIMA needs two calls, in order:** `fit_arima_model` then `forecast_time_series` —
  the fitted model is cached in-memory on the `DataManager` (`_arima_models`,
  keyed by `f"{dataset_id}:{column}"`), not persisted, so it's lost on process restart.
- **`pmdarima` is optional:** if it's not installed, `fit_arima_model` silently falls
  back to a fixed `(1,1,1)` order instead of auto-selecting one.
