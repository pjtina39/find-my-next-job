"""Predict article relevance using the Jev classifier via OpenRouter."""
import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from langchain_typesafe import Noul
from langchain_typesafe.types import NoulCriteria

from jev_openrouter import OpenRouterJevClassifier
from utils import log, params_show

load_dotenv()  # walks up from this file and finds the repo-root .env (OPENROUTER_API_KEY)


def load_articles(articles_path: str) -> pd.DataFrame:
    """Load articles from hive-partitioned parquet with separate title and content.

    Args:
        articles_path: Glob path to articles parquet files.

    Returns:
        DataFrame with columns: article-id, title, content, human_label.
    """
    return duckdb.connect().sql(f"""
        SELECT id AS "article-id", title, content, human_label
        FROM read_parquet('{articles_path}', hive_partitioning=true)
        WHERE content IS NOT NULL AND content != ''
    """).df()


def build_noul(prompt_path: str, prompt_source: str) -> Noul:
    """Build a Noul question from a prompt file.

    Args:
        prompt_path: Path to the prompt file (.txt for "svm", .json for "jev").
        prompt_source: "svm" to reuse existing tag prompts (stripped of {text} and other elements),
                       "jev" to use dedicated structured JSON prompts with criteria.

    Returns:
        Noul question instance.
    """
    path = Path(prompt_path)
    if prompt_source == "svm":
        text = path.read_text()
        # Strip the TEXT:\n\n{text} placeholder section
        if "\nTEXT:\n" in text:
            text = text.split("\nTEXT:\n")[0]
        # Adapt SVM prompt vocabulary to Noul question framing
        text = text.replace("INSTRUCTIONS:\n\nTag when:", "Valid when:")
        text = text.replace("Do not tag when:", "Not valid when:")
        text = text.replace("→ Tag", "→ Valid")
        text = text.replace("→ Don't tag", "→ Not valid")
        text = text.replace("→ tag", "→ Valid")
        text = text.replace("→ Don't Tag", "→ Not valid")
        text = text.replace("Confirmed positive examples:", "Valid examples:")
        text = text.replace("Confirmed negative examples:", "Not valid examples:")
        # Same closing question as jev-classifier, to keep the prompt as close as possible to the LLM prompt
        text = text.strip() + "\n\nIs TEXT valid according to one or more of the above criteria?"
        return Noul(instructions=text)
    elif prompt_source == "jev":
        config = json.loads(path.read_text())
        criteria = None
        if "criteria" in config:
            criteria = NoulCriteria(
                true=config["criteria"].get("true"),
                false=config["criteria"].get("false"),
            )
        return Noul(instructions=config["instructions"], criteria=criteria)
    else:
        raise ValueError(f"Unknown prompt_source: {prompt_source}")


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Logit transform: log(p / (1-p)) with clipping to avoid infinity."""
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))


def predict_batch(
    classifier: OpenRouterJevClassifier,
    articles: pd.DataFrame,
    noul: Noul,
    max_concurrency: int,
) -> pd.DataFrame:
    """Run Jev classifier on all articles and return predictions DataFrame.

    Args:
        classifier: Jev classifier instance.
        articles: DataFrame with columns [article-id, title, content, human_label].
        noul: Noul question instance (instructions + optional criteria).
        max_concurrency: Max concurrent API calls.

    Returns:
        DataFrame with columns [article-id, human_label, probability, decision_function, prediction,
        cost, input_tokens, output_tokens, latency_s], one row per article.
    """
    requests = [
        {
            "state": {"title": row["title"], "content": row["content"]},
            "questions": {"relevance": noul},
        }
        for _, row in articles.iterrows()
    ]

    log.info(f"Sending {len(requests)} articles to Jev (concurrency={max_concurrency})")
    start = time.time()
    responses = classifier.batch(
        requests,
        config={"max_concurrency": max_concurrency},
        return_exceptions=True,
    )
    elapsed = time.time() - start
    log.info(f"Jev batch completed in {elapsed:.1f}s ({elapsed / len(requests):.2f}s/article)")

    # One entry per article in each list; failed articles get NaN everywhere
    probabilities, costs, input_tokens, output_tokens, latencies = [], [], [], [], []
    n_failed = 0

    for i, resp in enumerate(responses):
        if isinstance(resp, Exception):
            log.warning(f"Article {i} failed: {resp}")
            for values in (probabilities, costs, input_tokens, output_tokens, latencies):
                values.append(np.nan)
            n_failed += 1
            continue

        # Extract probability from Noul response
        try:
            noul_result = resp.nouls["relevance"]
            prob = noul_result.noul
        except (AttributeError, KeyError, TypeError) as e:
            log.warning(f"Article {i}: could not extract probability: {e}")
            for values in (probabilities, costs, input_tokens, output_tokens, latencies):
                values.append(np.nan)
            n_failed += 1
            continue

        probabilities.append(prob)
        costs.append(resp.cost)
        input_tokens.append(resp.usage.input_tokens)
        output_tokens.append(resp.usage.output_tokens)
        latencies.append(resp.latency_s)

    if n_failed > 0:
        log.warning(f"{n_failed}/{len(responses)} articles failed Jev prediction")

    probs = np.array(probabilities)
    predictions = pd.DataFrame({
        "article-id": articles["article-id"].to_numpy(),
        "human_label": articles["human_label"].to_numpy(),
        "probability": probs,
        "decision_function": logit(probs),
        "prediction": (probs >= 0.5).astype(int),
        # dtype=float turns a missing (None) cost or token count into NaN
        "cost": np.array(costs, dtype=float),
        "input_tokens": np.array(input_tokens, dtype=float),
        "output_tokens": np.array(output_tokens, dtype=float),
        "latency_s": np.array(latencies, dtype=float),
    })

    log.info(f"Jev total cost: ${predictions['cost'].sum():.4f} "
             f"(input={int(predictions['input_tokens'].sum())}, "
             f"output={int(predictions['output_tokens'].sum())} tokens, "
             f"mean latency={predictions['latency_s'].mean():.2f}s/request)")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("articles", help="Glob path to articles parquet")
    parser.add_argument("output", help="Output path for predictions parquet")
    parser.add_argument("--prompt", required=True, help="Path to the prompt file")
    parser.add_argument("--prompt-source", choices=["svm", "jev"], default="svm", dest="prompt_source")
    parser.add_argument("--max-concurrency", type=int, default=8, dest="max_concurrency")
    args = parser.parse_args()

    articles_df = load_articles(args.articles)
    if articles_df.empty:
        log.error("No articles found")
        exit(1)
    log.info(f"Loaded {len(articles_df)} articles")

    noul = build_noul(args.prompt, args.prompt_source)
    log.info(f"Noul ({args.prompt_source}): {str(noul.instructions)[:100]}...")

    classifier = OpenRouterJevClassifier(timeout=60)
    predictions = predict_batch(classifier, articles_df, noul, args.max_concurrency)

    n_total = len(predictions)
    predictions = predictions.dropna(subset=["probability"])
    if len(predictions) < n_total:
        log.warning(f"Dropped {n_total - len(predictions)}/{n_total} articles with failed predictions")
    if predictions.empty:
        log.error(f"All {n_total} Jev requests failed; not writing {args.output}")
        exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.output)
    log.info(f"Saved {len(predictions)}/{n_total} predictions to {args.output}")


if __name__ == "__main__":
    main()
