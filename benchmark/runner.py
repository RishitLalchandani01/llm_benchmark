"""
Main runner / CLI.

    # everything in config.json
    python -m benchmark.runner

    # a local JSON question file
    python -m benchmark.runner --questions-file data/sample_questions.json \
                               --subject "AP Physics"

    # a PDF exam paper (figures are rendered and shown to vision models)
    python -m benchmark.runner --questions-file data/ap-physics-1.pdf \
                               --subject "AP Physics" --pages 3-12 \
                               --answer-key data/ap-physics-1.key.csv

    # transcribe a PDF to an editable JSON file and stop (no benchmarking)
    python -m benchmark.runner --questions-file data/ap-physics-1.pdf \
                               --subject "AP Physics" --extract-only

Each (model, question) pair is appended to results/raw_cache.jsonl as soon as it
is graded, so --resume can pick a crashed run back up without re-paying for the
answers you already have.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import pandas as pd

from .config import BenchmarkConfig, ModelConfig, QuestionSetConfig
from .dataset_loader import build_cli_question_set, load_all
from .evaluator import (
    accuracy_matrix,
    grade,
    overall_by_model,
    parse_answer,
    results_to_dataframe,
    summarize,
)
from .prompts import build_frq_messages, build_messages
from .schemas import MCQuestion, QuestionResult


def _text_limits(run_cfg) -> Dict[str, int]:
    """Truncation budgets, threaded to grade() so stored answers stay comparable."""
    return {
        "answer_text_chars": run_cfg.answer_text_chars,
        "raw_response_chars": run_cfg.raw_response_chars,
    }


# --------------------------------------------------------------------------- #
# One question, one model
# --------------------------------------------------------------------------- #
def run_single_question(
    client, model_cfg: ModelConfig, question: MCQuestion, judge=None
) -> QuestionResult:
    """
    Ask one question, parse it, grade it. Optionally re-asks once with a
    stricter prompt when the first reply is not parseable JSON.

    Figures are only attached for vision-capable models; text-only models are
    told the figure is missing and the row is flagged, so "could not see the
    diagram" stays distinguishable from "got the physics wrong".
    """
    if question.is_frq:
        return run_free_response(client, model_cfg, question, judge)

    attach = bool(model_cfg.supports_vision and question.image_paths)
    completion = client.complete(build_messages(question, attach_images=attach))

    # Hard API failure: record it as a wrong answer rather than dropping the row.
    if not completion.ok:
        return grade(
            question, None,
            model_name=model_cfg.name, provider=model_cfg.provider,
            params_b=model_cfg.params_b, latency_s=completion.latency_s,
            attempts=completion.attempts, status="api_error",
            error=completion.error or "empty completion", saw_image=attach,
            **_text_limits(client.run_cfg),
        )

    answer, reason = parse_answer(completion.text, question.letters)

    # Second chance with the "JSON ONLY" reminder appended.
    if answer is None and client.run_cfg.reask_on_parse_failure:
        retry = client.complete(build_messages(question, reask=True, attach_images=attach))
        if retry.ok:
            answer, reason = parse_answer(retry.text, question.letters)
            completion = retry

    return grade(
        question,
        answer.answer if answer else None,
        model_name=model_cfg.name,
        provider=model_cfg.provider,
        params_b=model_cfg.params_b,
        latency_s=completion.latency_s,
        attempts=completion.attempts,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        confidence=answer.confidence if answer else None,
        status="ok" if answer else "parse_error",
        error=None if answer else reason,
        raw_response=completion.text if client.run_cfg.save_raw_text else None,
        saw_image=attach,
        **_text_limits(client.run_cfg),
    )


def run_free_response(
    client, model_cfg: ModelConfig, question: MCQuestion, judge=None
) -> QuestionResult:
    """
    Free response: let the model write a full answer, then have the judge score
    it against the official guidelines. With no judge the answer is still
    collected and logged, but the row stays `ungraded`.
    """
    attach = bool(model_cfg.supports_vision and question.image_paths)
    completion = client.complete(build_frq_messages(question, attach_images=attach))

    if not completion.ok:
        return grade(
            question, None,
            model_name=model_cfg.name, provider=model_cfg.provider,
            params_b=model_cfg.params_b, latency_s=completion.latency_s,
            attempts=completion.attempts, status="api_error",
            error=completion.error or "empty completion", saw_image=attach,
            **_text_limits(client.run_cfg),
        )

    common = dict(
        model_name=model_cfg.name, provider=model_cfg.provider,
        params_b=model_cfg.params_b, latency_s=completion.latency_s,
        attempts=completion.attempts, prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens, saw_image=attach,
        answer_text=completion.text,
        raw_response=completion.text if client.run_cfg.save_raw_text else None,
        **_text_limits(client.run_cfg),
    )
    if judge is None:
        return grade(question, None, status="ungraded",
                     error="free response collected but no judge is enabled", **common)

    verdict = judge.score(question, completion.text)
    if not verdict.ok:
        return grade(question, None, status="ungraded",
                     error=f"judge failed: {verdict.error}", **common)

    return grade(
        question, None, status="ok",
        score=verdict.score, max_score=verdict.max_score,
        judge_model=judge.cfg.model, judge_reasoning=verdict.reasoning,
        **common,
    )


# --------------------------------------------------------------------------- #
# One model, all its questions
# --------------------------------------------------------------------------- #
def run_model(
    model_cfg: ModelConfig,
    questions: List[MCQuestion],
    cfg: BenchmarkConfig,
    cache_path: Optional[Path],
    already_done: Set[tuple],
    judge=None,
) -> List[QuestionResult]:
    """Run one model over its question list with a bounded thread pool."""
    from .llm_client import LLMClient, describe_vision, preflight  # lazy: --dry-run needs no litellm

    warning = preflight(model_cfg)
    if warning:
        print(f"  [warn] {model_cfg.name}: {warning}")

    pending = [q for q in questions if (model_cfg.name, q.question_id) not in already_done]
    skipped = len(questions) - len(pending)
    figures = sum(1 for q in pending if q.image_paths)
    print(f"\n>>> {model_cfg.name}: {len(pending)} to ask"
          f"{f' ({skipped} cached)' if skipped else ''}"
          f"{f' | {figures} with figures, {describe_vision(model_cfg)}' if figures else ''}")
    if not pending:
        return []

    client = LLMClient(model_cfg=model_cfg, run_cfg=cfg.run)
    results: List[QuestionResult] = []
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=cfg.run.max_workers) as pool:
        futures = {
            pool.submit(run_single_question, client, model_cfg, q, judge): q
            for q in pending
        }
        for i, future in enumerate(as_completed(futures), start=1):
            question = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # belt and braces; run_single_question shouldn't raise
                result = grade(
                    question, None, model_name=model_cfg.name,
                    provider=model_cfg.provider, params_b=model_cfg.params_b,
                    status="api_error", error=f"worker crash: {exc}"[:300],
                )
            results.append(result)
            if cache_path:
                _append_cache(cache_path, result)
            if i % 10 == 0 or i == len(pending):
                scored = [r for r in results if r.graded]
                credit = sum(
                    (r.score / r.max_score) if (r.score is not None and r.max_score)
                    else float(r.is_correct)
                    for r in scored
                )
                pct = 100 * credit / len(scored) if scored else float("nan")
                print(f"    {i}/{len(pending)} done | running credit {pct:.1f}% "
                      f"({len(scored)} scored)")

    print(f"    finished {model_cfg.name} in {time.perf_counter() - started:.1f}s")
    return results


# --------------------------------------------------------------------------- #
# Resume cache
# --------------------------------------------------------------------------- #
def _append_cache(path: Path, result: QuestionResult) -> None:
    """Append-only JSONL so a crash never loses completed work."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result.model_dump(), default=str) + "\n")


def _load_cache(path: Path) -> List[QuestionResult]:
    if not path.exists():
        return []
    out: List[QuestionResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(QuestionResult.model_validate(json.loads(line)))
        except Exception:
            continue  # ignore a half-written final line
    return out


# --------------------------------------------------------------------------- #
# Question-set selection (config.json sets vs. command-line files)
# --------------------------------------------------------------------------- #
def resolve_question_sets(
    cfg: BenchmarkConfig,
    set_names: Optional[Sequence[str]],
    questions_files: Optional[Sequence[str]],
    subjects: Optional[Sequence[str]],
    *,
    input_format: str = "auto",
    pages: Optional[str] = None,
    pdf_engine: Optional[str] = None,
    answer_key: Optional[str] = None,
    include_frq: bool = False,
    limit: Optional[int] = None,
) -> List[QuestionSetConfig]:
    """
    --questions-file wins over config.json's question_sets.

    One --subject applies to every file; otherwise give one --subject per file,
    in the same order.
    """
    if not questions_files:
        return cfg.select_question_sets(list(set_names) if set_names else None)

    subjects = list(subjects or [])
    if not subjects:
        raise SystemExit("--questions-file requires --subject (e.g. --subject \"AP Physics\")")
    if len(subjects) == 1:
        subjects = subjects * len(questions_files)
    elif len(subjects) != len(questions_files):
        raise SystemExit(
            f"got {len(questions_files)} --questions-file but {len(subjects)} --subject; "
            f"pass one subject, or one per file in the same order"
        )

    return [
        build_cli_question_set(
            path, subject,
            input_format=input_format, pages=pages, pdf_engine=pdf_engine,
            answer_key=answer_key, include_frq=include_frq, limit=limit,
        )
        for path, subject in zip(questions_files, subjects)
    ]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_benchmark(
    config_path: str,
    model_names: Optional[List[str]] = None,
    set_names: Optional[List[str]] = None,
    limit: Optional[int] = None,
    resume: bool = False,
    dry_run: bool = False,
    *,
    questions_files: Optional[List[str]] = None,
    subjects: Optional[List[str]] = None,
    input_format: str = "auto",
    pages: Optional[str] = None,
    pdf_engine: Optional[str] = None,
    answer_key: Optional[str] = None,
    include_frq: bool = False,
    dpi: Optional[int] = None,
    attach_figures: Optional[str] = None,
    refresh_extract: bool = False,
    extract_only: bool = False,
    extract_out: Optional[str] = None,
    judge_enabled: Optional[bool] = None,
    judge_model: Optional[str] = None,
    rubric: Optional[str] = None,
    max_score: Optional[float] = None,
    compare: bool = False,
    bootstrap: int = 10_000,
) -> Dict[str, pd.DataFrame]:
    cfg = BenchmarkConfig.load(config_path)

    # ---- judge wiring (free-response grading) ----------------------------- #
    if judge_enabled is not None:
        cfg.judge.enabled = judge_enabled
    if judge_model:
        cfg.judge.model = judge_model
        cfg.judge.enabled = True
    if rubric:
        cfg.judge.rubric_path = rubric
        cfg.judge.enabled = True
    if max_score is not None:
        cfg.judge.default_max_score = max_score
    if cfg.judge.enabled and not include_frq:
        include_frq = True          # judging implies you want the FRQ items

    # Command-line overrides for the PDF pipeline.
    if dpi:
        cfg.pdf.dpi = dpi
    if attach_figures:
        cfg.pdf.attach_figures = attach_figures

    sets = resolve_question_sets(
        cfg, set_names, questions_files, subjects,
        input_format=input_format, pages=pages, pdf_engine=pdf_engine,
        answer_key=answer_key, include_frq=include_frq, limit=limit,
    )

    # ---- transcribe a PDF and stop ---------------------------------------- #
    if extract_only:
        from .pdf_loader import extract_to_json

        pdf_sets = [s for s in sets if s.format == "pdf"]
        if not pdf_sets:
            raise SystemExit("--extract-only needs a PDF: pass --questions-file <file>.pdf")
        for s in pdf_sets:
            default_out = Path(s.local_path or "").with_suffix(".questions.json")
            extract_to_json(
                s.local_path or "",
                extract_out or default_out,
                set_name=s.name, domain=s.domain,
                pdf_cfg=cfg.pdf, run_cfg=cfg.run,
                pages=s.pages, engine=s.pdf_engine, refresh=refresh_extract,
            )
        return {}

    models = cfg.select_models(model_names)
    out_dir = Path(cfg.run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / cfg.run.cache_jsonl

    print("=" * 72)
    print(f"models        : {', '.join(m.name for m in models)}")
    print(f"question sets : {', '.join(f'{s.name} [{s.format}] -> {s.domain}' for s in sets)}")
    if cfg.judge.enabled:
        print(f"free-response : judged by {cfg.judge.model}"
              f"{f' against {cfg.judge.rubric_path}' if cfg.judge.rubric_path else ' with no rubric'}")
    print("=" * 72)

    print("\nloading questions...")
    by_set = load_all(
        sets, limit_override=limit, seed=cfg.run.seed,
        shuffle=cfg.run.shuffle_questions,
        pdf_cfg=cfg.pdf, run_cfg=cfg.run, refresh_extract=refresh_extract,
    )
    questions = [q for qs in by_set.values() for q in qs]
    if not questions:
        raise SystemExit("no questions loaded -- check your question sets / PDF pages")

    n_gradeable = sum(1 for q in questions if q.is_gradeable)
    n_frq = sum(1 for q in questions if q.is_frq)
    n_figures = sum(1 for q in questions if q.image_paths)
    total_calls = len(questions) * len(models) * (2 if (n_frq and cfg.judge.enabled) else 1)
    print(f"\n{len(questions)} questions ({n_gradeable} multiple-choice with a key, "
          f"{n_frq} free-response, {n_figures} with figures) x {len(models)} models "
          f"= up to {total_calls} API calls"
          f"{' (free-response answers are judged, so 2 calls each)' if n_frq and cfg.judge.enabled else ''}")
    if n_gradeable == 0 and not (n_frq and cfg.judge.enabled):
        print("WARNING: nothing here can be scored. Supply --answer-key for "
              "multiple choice, or --judge (with --rubric) for free response.")
    if n_frq and not cfg.judge.enabled:
        print("NOTE: free-response questions are present but no judge is enabled; "
              "their answers will be collected and left ungraded. Add --judge.")
    if dry_run:
        print("\n--dry-run: stopping before any benchmark API call.")
        return {}

    # Resume: replay what we already paid for.
    prior: List[QuestionResult] = _load_cache(cache_path) if resume else []
    done: Set[tuple] = {(r.model_name, r.question_id) for r in prior}
    if prior:
        print(f"resuming: {len(prior)} cached results loaded from {cache_path}")
    elif cache_path.exists() and not resume:
        cache_path.unlink()  # fresh run -> fresh cache

    judge = None
    if cfg.judge.enabled and n_frq:
        from .judge import Judge

        judge = Judge(cfg.judge, cfg.run)

    results: List[QuestionResult] = list(prior)
    for model_cfg in models:
        results.extend(run_model(model_cfg, questions, cfg, cache_path, done, judge))

    # ------------------------------ reporting ------------------------------ #
    raw = results_to_dataframe(results)
    summary = summarize(raw)
    overall = overall_by_model(summary)
    matrix = accuracy_matrix(summary)

    raw_path = out_dir / cfg.run.raw_csv
    summary_path = out_dir / cfg.run.summary_csv
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    overall.to_csv(out_dir / "overall_by_model.csv", index=False)
    matrix.to_csv(out_dir / "accuracy_matrix.csv")

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\n" + "=" * 72)
    print("PER-MODEL / PER-DOMAIN SUMMARY")
    print("=" * 72)
    print(summary.to_string(index=False))
    print("\nSCORE MATRIX (credit %: MCQ correctness and judged free-response points)")
    print(matrix.to_string())
    print("\nOVERALL (macro-average across domains)")
    print(overall.to_string(index=False))
    print(f"\nwrote:\n  {raw_path}\n  {summary_path}\n"
          f"  {out_dir / 'overall_by_model.csv'}\n  {out_dir / 'accuracy_matrix.csv'}")

    bundle = {"raw": raw, "summary": summary, "overall": overall, "matrix": matrix}

    # Cross-model comparison: paired per-item analysis over the identical paper.
    # Re-runnable at any time from the CSV alone -- `python -m benchmark.compare`.
    if compare and len(models) > 1:
        from .compare import compare_all

        print("\n" + "=" * 72)
        print("CROSS-MODEL COMPARISON")
        print("=" * 72)
        bundle["comparison"] = compare_all(
            raw, out_dir / "comparison", n_boot=bootstrap, seed=cfg.run.seed
        )
    elif compare:
        print("\n[compare] skipped: a comparison needs at least two models")

    return bundle


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmark.runner",
        description="Benchmark several LLMs on SAT / AP multiple-choice questions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m benchmark.runner --list\n"
            "  python -m benchmark.runner --questions-file data/exam.json --subject \"AP Math\"\n"
            "  python -m benchmark.runner --questions-file data/ap-physics-1.pdf \\\n"
            "                             --subject \"AP Physics\" --pages 3-12 --extract-only\n"
        ),
    )
    p.add_argument("--config", default="config.json", help="path to config.json")
    p.add_argument("--models", nargs="*", default=None,
                   help="model names from config.json (default: all enabled)")
    p.add_argument("--sets", nargs="*", default=None, dest="sets",
                   help="question set names from config.json (default: all enabled)")
    p.add_argument("--limit", type=int, default=None,
                   help="override questions-per-set (useful for smoke tests)")
    p.add_argument("--resume", action="store_true",
                   help="reuse results already in results/raw_cache.jsonl")
    p.add_argument("--dry-run", action="store_true",
                   help="load questions and print the plan, run no benchmark calls")
    p.add_argument("--list", action="store_true",
                   help="list configured models and question sets, then exit")

    src = p.add_argument_group("question source (overrides config.json sets)")
    src.add_argument("--questions-file", action="append", default=None, metavar="PATH",
                     help="a .json/.jsonl question file or a .pdf exam paper; repeatable")
    src.add_argument("--subject", action="append", default=None, metavar="NAME",
                     help='subject/domain label, e.g. --subject "AP Physics". One for '
                          "all files, or one per file in the same order")
    src.add_argument("--input-format", choices=["auto", "json", "pdf"], default="auto",
                     help="force the file type instead of guessing from the extension")
    src.add_argument("--answer-key", default=None, metavar="SPEC",
                     help="answer key: a .json/.csv/.txt file, or inline \"1:B,2:C\"")
    src.add_argument("--include-frq", action="store_true",
                     help="keep free-response questions (asked, but reported ungraded)")

    pdf = p.add_argument_group("PDF options")
    pdf.add_argument("--pages", default=None, metavar="SPEC",
                     help='1-based page spec, e.g. "3-12" or "2,5,7-9" (default: all)')
    pdf.add_argument("--pdf-engine", choices=["auto", "text", "vision"], default=None,
                     help="text = regex the text layer (free); vision = transcribe page "
                          "images with an LLM (handles figures and real notation)")
    pdf.add_argument("--dpi", type=int, default=None,
                     help="page render resolution (default from config.json, 150)")
    pdf.add_argument("--attach-figures", choices=["auto", "always", "never"], default=None,
                     help="send rendered page images to vision-capable models")
    pdf.add_argument("--extract-only", action="store_true",
                     help="transcribe the PDF to an editable JSON file and stop")
    pdf.add_argument("--extract-out", default=None, metavar="PATH",
                     help="where --extract-only writes (default: <pdf>.questions.json)")
    pdf.add_argument("--refresh-extract", action="store_true",
                     help="ignore the cached extraction and re-read the PDF")

    jg = p.add_argument_group("free-response grading (LLM judge)")
    jg.add_argument("--judge", dest="judge_enabled", action="store_true", default=None,
                    help="score free-response answers with the judge model "
                         "(implies --include-frq)")
    jg.add_argument("--no-judge", dest="judge_enabled", action="store_false",
                    help="disable the judge even if config.json enables it")
    jg.add_argument("--judge-model", default=None, metavar="MODEL",
                    help="litellm id of the grader, e.g. gemini/gemini-2.5-pro. "
                         "Use a model that is NOT under test")
    jg.add_argument("--rubric", default=None, metavar="PATH",
                    help="official scoring guidelines: .pdf, .txt/.md, or a "
                         '.json mapping {"1": "rubric text"}')
    jg.add_argument("--max-score", type=float, default=None, metavar="N",
                    help="points per free-response question when the rubric "
                         "does not state a total (default 1.0)")

    cmp_ = p.add_argument_group("cross-model comparison")
    cmp_.add_argument("--compare", action="store_true",
                      help="after the run, compare how the models answered the same "
                           "paper item by item: pairwise McNemar tests, Wilson "
                           "intervals, marginal utility per parameter decade, and the "
                           "most discriminating items. Also available standalone: "
                           "python -m benchmark.compare results/raw_responses.csv")
    cmp_.add_argument("--bootstrap", type=int, default=10_000, metavar="B",
                      help="bootstrap resamples for marginal-utility intervals "
                           "(default: %(default)s)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.list:
        cfg = BenchmarkConfig.load(args.config)
        print("models:")
        for m in cfg.models:
            flags = [] if m.enabled else ["disabled"]
            if m.supports_vision:
                flags.append("vision")
            size = f"{m.params_b}B" if m.params_b else "undisclosed"
            suffix = f"  ({', '.join(flags)})" if flags else ""
            print(f"  {m.name:<22} {m.litellm_model:<34} {size}{suffix}")
        print("question sets:")
        for q in cfg.question_sets:
            flag = "" if q.enabled else "  (disabled)"
            print(f"  {q.name:<22} {q.domain:<20} [{q.format}] "
                  f"{q.hf_path or q.local_path} limit={q.limit}{flag}")
        print(f"pdf: engine={cfg.pdf.engine} dpi={cfg.pdf.dpi} "
              f"extractor={cfg.pdf.extractor_model} cache={cfg.pdf.cache_dir}")
        print(f"judge: enabled={cfg.judge.enabled} model={cfg.judge.model} "
              f"rubric={cfg.judge.rubric_path} max_score={cfg.judge.default_max_score}")
        return 0

    try:
        run_benchmark(
            config_path=args.config,
            model_names=args.models,
            set_names=args.sets,
            limit=args.limit,
            resume=args.resume,
            dry_run=args.dry_run,
            questions_files=args.questions_file,
            subjects=args.subject,
            input_format=args.input_format,
            pages=args.pages,
            pdf_engine=args.pdf_engine,
            answer_key=args.answer_key,
            include_frq=args.include_frq,
            dpi=args.dpi,
            attach_figures=args.attach_figures,
            refresh_extract=args.refresh_extract,
            extract_only=args.extract_only,
            extract_out=args.extract_out,
            judge_enabled=args.judge_enabled,
            judge_model=args.judge_model,
            rubric=args.rubric,
            max_score=args.max_score,
            compare=args.compare,
            bootstrap=args.bootstrap,
        )
    except KeyboardInterrupt:
        print("\ninterrupted -- rerun with --resume to continue from the cache")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
