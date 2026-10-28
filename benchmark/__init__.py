"""
benchmark
=========
A small, modular framework for scoring several LLMs on standardized academic
tests (SAT, AP Math / Physics / Philosophy), from Hugging Face datasets, your
own JSON files, or PDF exam papers.

Module map
----------
config.py          -> typed, JSON-driven configuration (no hardcoded secrets)
schemas.py         -> pydantic models for questions, answers and results
prompts.py         -> prompt construction: MCQ (JSON answer), free response, judge
dataset_loader.py  -> Hugging Face / JSON loaders normalised to MCQuestion
pdf_loader.py      -> PDF exam papers: text or vision extraction, figures, keys
llm_client.py      -> litellm wrapper: retries, backoff, rate limiting, images
judge.py           -> LLM-as-judge scoring for free-response questions
evaluator.py       -> answer parsing, grading, aggregation to DataFrames
compare.py         -> paired cross-model analysis of the same paper (also a CLI)
runner.py          -> CLI entry point that wires everything together
"""

__version__ = "2.1.0"
