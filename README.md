# LLM Scaling Benchmark (SAT / AP)

Does exponentially more parameters actually buy exponentially more utility?
This framework scores several LLMs on the same standardized multiple-choice
questions (SAT Math, SAT Reading, AP Math, AP Physics, AP Philosophy) and
reports accuracy, latency and an accuracy-per-billion-parameters figure.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then fill in your keys
set -a && source .env && set +a

# local Llamas
ollama pull llama3.2:1b && ollama pull llama3.2:3b && ollama pull llama3.1:8b
```

## Run

```bash
python -m benchmark.runner --list                       # show configured models/sets
python -m benchmark.runner --dry-run                    # cost-free plan + question counts
python -m benchmark.runner --models llama-3.2-1b llama-3.2-3b llama-3.1-8b \
                           --sets sat_math ap_physics --limit 25
python -m benchmark.runner --resume                     # continue an interrupted run
```

## Your own questions: a JSON file or a PDF

`--questions-file` takes either, and `--subject` is the domain label that ends up
in the report. Both flags repeat, so several papers can run in one go.

```bash
# a JSON file you wrote or edited by hand
python -m benchmark.runner --questions-file data/sample_questions.json \
                           --subject "AP Physics"

# a PDF exam paper, pages 3-12, with an answer key
python -m benchmark.runner --questions-file data/ap-physics-1.pdf \
                           --subject "AP Physics" --pages 3-12 \
                           --answer-key data/ap-physics-1.key.csv

# two papers, two subjects, one run
python -m benchmark.runner --questions-file data/sat.pdf --subject "SAT Math" \
                           --questions-file data/apphys.pdf --subject "AP Physics"
```

### The recommended PDF workflow

An exam paper almost never ships with its answer key, and no extractor is
perfect, so transcribe first and check the result:

```bash
# 1. transcribe to an editable JSON file (no benchmarking, no grading)
python -m benchmark.runner --questions-file data/ap-physics-1.pdf \
                           --subject "AP Physics" --extract-only

# 2. open data/ap-physics-1.questions.json, fix any garbled text, and fill in
#    each "answer": "B"

# 3. benchmark against the file you just checked
python -m benchmark.runner --questions-file data/ap-physics-1.questions.json \
                           --subject "AP Physics"
```

### PDF flags

| flag | meaning |
|---|---|
| `--pages 3-12` | 1-based page spec; also `2,5,7-9`. Default: all pages |
| `--pdf-engine text` | regex the PDF's text layer. Free and instant, but mangles real notation (`h2 > h1` comes out as `h h > 2 1`) |
| `--pdf-engine vision` | render each page and have a vision model transcribe it. Costs ~1 call per page, survives figures and equations |
| `--pdf-engine auto` | text first, vision for pages the text parser fails on (default) |
| `--answer-key SPEC` | a `.json`/`.csv`/`.txt` file, or inline `"1:B,2:C"` |
| `--include-frq` | keep free-response items (asked, but reported `ungraded`) |
| `--dpi 200` | page render resolution |
| `--attach-figures never` | stop sending page images even to vision models |
| `--refresh-extract` | ignore the cached extraction and re-read the PDF |

Extractions are cached in `data/cache/`, so a given PDF costs extraction once.

### Figures

Exam figures are usually vector drawings, so the whole page is rendered to PNG
and attached to the questions that need it. Only models with
`"supports_vision": true` in `config.json` are sent the image; text-only models
are told the figure is missing, and `summary.csv` reports `figure_questions` vs
`figures_sent` so you can see which comparisons were fair.

## Free-response questions (LLM judge)

Most released AP papers are free response: "derive an expression for...",
"justify your answer...". There is no letter to compare, so a judge model reads
each answer against the official scoring guidelines and awards points.

```bash
python -m benchmark.runner --questions-file data/ap-physics-1.pdf \
                           --subject "AP Physics" \
                           --judge --judge-model gemini/gemini-2.5-pro \
                           --rubric data/ap-physics-1-scoring-guidelines.pdf
```

`--judge` implies `--include-frq`. Each free-response item then costs two calls:
one for the model under test to write its answer, one for the judge to score it.

| flag | meaning |
|---|---|
| `--judge` / `--no-judge` | enable or disable judged grading |
| `--judge-model MODEL` | the grader's litellm id. Use a model that is **not** under test |
| `--rubric PATH` | scoring guidelines: `.pdf`, `.txt`/`.md`, or `.json` as `{"1": "rubric text for question 1"}` |
| `--max-score N` | points per question when the rubric states no total (default 1.0) |

A `.pdf`/`.txt` rubric is split on `Question N` headings so each question is
judged against its own section, and `"7 points total"` in a section sets that
question's `max_score` automatically.

### Reading the scores

`summary.csv` keeps the two question types apart, because they are not the same
kind of measurement:

* `accuracy_pct` — multiple choice only, exact letter match against the key.
* `frq_score_pct` — judged points earned / points available.
* `credit_pct` — headline figure over both: a 3-of-4-point answer counts as 0.75
  of an item. Equals plain accuracy in a run with no free response.

The judge's own reasoning is written to `raw_responses.csv` (`judge_reasoning`,
`answer_text`), so every score can be spot-checked by hand.

**Be honest about this in the paper:** a judged score is one model's opinion, not
ground truth. Spot-check a sample yourself, say which judge you used, and note
that the judge was not one of the models under test.

### Questions with no answer key

They are still asked and logged, but reported as `ungraded` and excluded from the
accuracy denominator, never silently counted as wrong. `summary.csv` splits
`n_questions` into `n_graded` + `n_ungraded`.

## Output (`results/`)

| file | contents |
|---|---|
| `raw_responses.csv` | one row per (model, question): prediction, correctness, latency, tokens, raw text |
| `summary.csv` | per model x domain: accuracy %, avg/p95 latency, parse failures, API errors |
| `accuracy_matrix.csv` | models x domains pivot of accuracy % |
| `overall_by_model.csv` | macro-average accuracy and accuracy per billion parameters |
| `raw_cache.jsonl` | append-only log powering `--resume` |

## Comparing models on the same paper

Every model answers the *identical* item set in the identical order (one seeded
permutation, reused for each model), so the design is paired. `--compare` exploits
that: it joins the per-item rows back together and reports what the aggregate
tables cannot.

```bash
# as part of a run
python -m benchmark.runner --limit 300 --compare

# or standalone, on a CSV you already paid for -- no API calls
python -m benchmark.compare results/raw_responses.csv
python -m benchmark.compare results/raw_responses.csv --bootstrap 10000 --top-items 40
```

Output lands in `results/comparison/`:

| file | what it answers |
|---|---|
| `comparison_by_question.csv` | one row per question, one column per model holding the chosen letter (or the rubric fraction) — the literal side-by-side view |
| `pairwise_models.csv` | for each model pair: both right / both wrong / A-only / B-only, agreement, letter-level Cohen's κ, **McNemar exact test** with Holm–Bonferroni adjustment |
| `accuracy_with_intervals.csv` | per-domain accuracy with **Wilson 95% intervals** |
| `marginal_utility.csv` | accuracy points gained **per decade of parameters**, with a stratified item bootstrap interval |
| `marginal_utility_by_format.csv` | the same intervals computed separately for multiple choice and free response, plus their ratio |
| `discriminating_items.csv` | items that split the suite, and **scale inversions** — a larger model wrong where a smaller one is right |
| `calibration.csv` | accuracy within confidence bands, per model, with expected calibration error |
| `frq_comparison.md` | every model's free-response answer side by side with the judge's score and reasoning |

Three things worth knowing:

* **McNemar runs on multiple-choice items only.** The test is defined for paired
  binary outcomes; rubric-scored partial credit is not binary. Wilson intervals are
  likewise reported only where accuracy is a genuine binomial proportion.
* **Pairing is verified, not assumed.** If models answered different item sets —
  which happens if you change `--limit` between runs and then `--resume` — you get a
  loud warning and the statistics fall back to the common items.
* **Models with no disclosed parameter count are excluded from anything per-parameter.**
  Gemini and Grok sizes are not public; an interval needs two abscissae, and imputing
  one would put a guess inside the headline quantity.

## Notes on methodology

* `temperature: 0.0` and a fixed `seed` for question sampling, so every model
  sees the identical questions in the identical order.
* API failures are scored as **incorrect** rather than dropped, so an unreliable
  model cannot inflate its accuracy by only answering what it managed to reach.
* Parse failures are counted separately in `summary.csv` — for small models this
  is itself a utility signal (a model that cannot follow a format is less useful
  even when it "knows" the answer).
