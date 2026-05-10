import modal
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / "Data"
LOCAL_OUTPUT_DIR = SCRIPT_DIR.parent.parent / "Results" / "LLM_as_judge"
ENV_FILE = SCRIPT_DIR.parent.parent.parent / ".env"

TOML_PATH = "/data/datasets.toml"
INPUT_DIR = "/data/input"
OUTPUT_DIR = "/data/output"

MAX_CONCURRENCY_LLM = 12
LLM_BATCH_SIZE = 50
PREFILTER_BATCH_SIZE = 128
MAX_PREFILTER_ROWS = 10000
EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBED_DIM = 3072
EXCLUDED_DATASETS = {"Chou_2004", "Jeyaraman_2020", "Smid_2020", "Moran_2021"}

ROW_PREFIX = "[[<ROW_ID::"
ROW_SUFFIX = ">]]"

DECISION_COLUMNS = [
    "dataset",
    "permanent_row_number",
    "permanent_row_id",
    "title",
    "abstract",
    "label_included",
    "decision",
    "decision_reason",
    "llm_response",
    "rank_position",
    "cross_encoding_score",
    "cosine_score",
    "prefiltered",
]

results_volume = modal.Volume.from_name("llm_as_judge_simple", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pandas",
        "openai",
        "python-dotenv",
        "scikit-learn",
        "numpy",
    )
    .add_local_dir(
        str(DATA_DIR / "cleaned_synergy_dataset"),
        remote_path="/data/input",
    )
    .add_local_file(
        str(DATA_DIR / "datasets.toml"),
        remote_path="/data/datasets.toml",
    )
)

app = modal.App("llm-as-judge-simple")


class LLMAbstractScreeningExperiment:

    def __init__(self, llm_client, embedding_model=EMBEDDING_MODEL):
        self.llm_client = llm_client
        self.embedding_model = embedding_model

    def get_criteria_text(self, toml_path, dataset_key):
        import tomllib

        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

        for d in data["datasets"]:
            if d["key"] == dataset_key:
                pub = d.get("publication", {})
                return f"{pub.get('title','')}\n{pub.get('eligibility_criteria','')}"
        return ""

    def load_dataset(self, dataset_key):
        path = os.path.join(INPUT_DIR, f"{dataset_key}.csv")
        return pd.read_csv(path)

    def assign_permanent_ids(self, dataset):
        dataset = dataset.copy()
        dataset["permanent_row_number"] = dataset.index
        dataset["permanent_row_id"] = "MATCH_ROW_" + dataset["permanent_row_number"].astype(str)
        return dataset

    def add_row_tag(self, document_text, row_id):
        return f"{ROW_PREFIX}{row_id}{ROW_SUFFIX}\n{document_text}"

    async def make_llm_call(self, user_prompt, system_prompt, row_id, valid_responses={0, 1, 2}):
        try:
            response = await self.llm_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                reasoning_effort="medium"
            )
            content = response.choices[0].message.content
            if int(str(content).strip()) not in valid_responses:
                print(f"Invalid response for {row_id}: {content}. RETRYING...")
                return await self.make_llm_call(user_prompt, system_prompt, row_id, valid_responses)
        except Exception as exc:
            print(f"LLM request failed for {row_id}: {exc}. RETRYING...")
            return await self.make_llm_call(user_prompt, system_prompt, row_id, valid_responses)
        return {"row_id": row_id, "llm_response": int(str(content).strip())}

    async def run_llm_batch(self, system_prompt, criteria, batch_ids, doc_map, sem):
        import asyncio

        async def bounded_call(rid):
            async with sem:
                doc = doc_map[rid]
                user_prompt = f"Criteria: {criteria}\n\nAbstract and Title:\n{doc}"
                return await self.make_llm_call(user_prompt, system_prompt, rid)

        tasks = [asyncio.create_task(bounded_call(rid)) for rid in batch_ids]
        return await asyncio.gather(*tasks)

    async def embed_texts(self, texts, batch_size=PREFILTER_BATCH_SIZE):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            clean = [str(x).replace("\n", " ").strip() or "empty" for x in batch]
            try:
                resp = await self.llm_client.embeddings.create(
                    input=clean, model=self.embedding_model
                )
                embeddings.extend([d.embedding for d in resp.data])
            except Exception as exc:
                print("Embedding call failed", exc)
                embeddings.extend([[0.0] * DEFAULT_EMBED_DIM for _ in batch])
        return embeddings

    async def compute_prefilter_scores(self, texts, criteria_text):
        if not texts:
            return []

        anchor_resp = await self.llm_client.embeddings.create(
            input=[criteria_text], model=self.embedding_model
        )
        anchor_vec = np.array(anchor_resp.data[0].embedding)
        anchor_norm = np.linalg.norm(anchor_vec) or 1.0

        doc_embeddings = await self.embed_texts(texts)
        matrix = np.array(doc_embeddings)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_docs = matrix / norms
        normalized_anchor = anchor_vec / anchor_norm
        cosines = normalized_docs.dot(normalized_anchor)
        return cosines.tolist()

    async def run_experiment(self, dataset_key, system_prompt):
        dataset = self.load_dataset(dataset_key)
        if dataset.empty:
            empty_summary = {
                "dataset": dataset_key,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "workload_reduction": 0.0,
                "total_rows": 0,
                "subset_rows": 0,
                "prefiltered_rows": 0,
            }
            return empty_summary, pd.DataFrame(columns=DECISION_COLUMNS)

        dataset = self.assign_permanent_ids(dataset)
        criteria = self.get_criteria_text(TOML_PATH, dataset_key)
        dataset["combined_text"] = (
            dataset["title"].fillna("") + " " + dataset["abstract"].fillna("")
        )

        keep_mask = np.ones(len(dataset), dtype=bool)
        prefiltered_ids = []
        if len(dataset) > MAX_PREFILTER_ROWS:
            try:
                cosine_scores = await self.compute_prefilter_scores(
                    dataset["combined_text"].tolist(), criteria
                )
            except Exception:
                cosine_scores = [0.0] * len(dataset)
            dataset["cosine_score"] = cosine_scores
            keep_mask[:] = False
            kth = max(len(cosine_scores) - MAX_PREFILTER_ROWS, 0)
            top_indices = np.argpartition(cosine_scores, kth)[kth:]
            keep_mask[top_indices] = True
            prefiltered_ids = dataset.loc[~keep_mask, "permanent_row_id"].tolist()
        else:
            dataset["cosine_score"] = np.nan

        dataset["prefiltered"] = ~keep_mask
        subset = dataset.loc[keep_mask].copy()
        subset.drop(columns=["combined_text"], inplace=True, errors="ignore")
        dataset.drop(columns=["combined_text"], inplace=True, errors="ignore")

        if subset.empty:
            empty_summary = {
                "dataset": dataset_key,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "workload_reduction": 1.0,
                "total_rows": len(dataset),
                "subset_rows": 0,
                "prefiltered_rows": len(dataset),
            }
            log = pd.DataFrame(columns=DECISION_COLUMNS)
            return empty_summary, log

        docs = []
        ids = []
        for rid, title, abstract in zip(
            subset["permanent_row_id"], subset["title"].fillna(""), subset["abstract"].fillna("")
        ):
            text = f"Title: {title}\nAbstract: {abstract}"
            docs.append(self.add_row_tag(text, rid))
            ids.append(str(rid))

        doc_map = dict(zip(ids, docs))
        sem = __import__("asyncio").Semaphore(MAX_CONCURRENCY_LLM)
        results = await self.run_llm_batch(system_prompt, criteria, ids, doc_map, sem)

        decisions = {rid: 0 for rid in dataset["permanent_row_id"]}
        reason_map = {rid: "default_exclude" for rid in dataset["permanent_row_id"]}
        llm_responses = {}
        rank_positions = {}

        for rid in prefiltered_ids:
            reason_map[rid] = "prefilter_cosine_top10000"

        for idx, result in enumerate(results):
            rid = result["row_id"]
            response = result["llm_response"]
            decisions[rid] = 1 if response in [1, 2] else 0
            reason_map[rid] = f"llm_response_{response}"
            llm_responses[rid] = response
            rank_positions[rid] = idx

        final_predictions = [decisions[rid] for rid in dataset["permanent_row_id"]]
        truth = dataset.set_index("permanent_row_id")["label_included"].astype(int)
        true_labels = [int(truth[rid]) for rid in dataset["permanent_row_id"]]

        accuracy = accuracy_score(true_labels, final_predictions)
        precision = precision_score(true_labels, final_predictions, zero_division=0)
        recall = recall_score(true_labels, final_predictions, zero_division=0)
        f1 = f1_score(true_labels, final_predictions, zero_division=0)
        workload_reduction = final_predictions.count(0) / len(final_predictions)

        summary = {
            "dataset": dataset_key,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "workload_reduction": workload_reduction,
            "total_rows": len(dataset),
            "subset_rows": len(subset),
            "prefiltered_rows": len(prefiltered_ids),
            "llm_calls": len(results),
        }

        log_rows = []
        for _, row in dataset.iterrows():
            rid = row["permanent_row_id"]
            log_rows.append(
                {
                    "dataset": dataset_key,
                    "permanent_row_number": int(row["permanent_row_number"]),
                    "permanent_row_id": rid,
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "label_included": int(row["label_included"]),
                    "decision": int(decisions[rid]),
                    "decision_reason": reason_map.get(rid, "unknown"),
                    "llm_response": llm_responses.get(rid),
                    "rank_position": rank_positions.get(rid),
                    "cross_encoding_score": None,
                    "cosine_score": float(row.get("cosine_score", np.nan))
                    if not pd.isna(row.get("cosine_score", np.nan))
                    else np.nan,
                    "prefiltered": bool(row["prefiltered"]),
                }
            )

        return summary, pd.DataFrame(log_rows)[DECISION_COLUMNS]

def decision_log_path(dataset_key: str) -> Path:
    return Path(OUTPUT_DIR) / f"{dataset_key}_decisions.csv"


def gather_decision_logs() -> pd.DataFrame:
    logs = []
    for path in sorted(Path(OUTPUT_DIR).glob("*_decisions.csv")):
        try:
            logs.append(pd.read_csv(path))
        except Exception as exc:
            print(f"Unable to read {path}: {exc}")
    if not logs:
        return pd.DataFrame(columns=DECISION_COLUMNS)
    return pd.concat(logs, ignore_index=True)


def compute_dataset_metrics(all_decisions: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    for dataset_key, group in all_decisions.groupby("dataset", sort=True):
        preds = group["decision"].astype(int)
        truths = group["label_included"].astype(int)
        metrics.append({
            "dataset": dataset_key,
            "num_papers": len(group),
            "precision": precision_score(truths, preds, zero_division=0),
            "recall": recall_score(truths, preds, zero_division=0),
            "f1": f1_score(truths, preds, zero_division=0),
            "workload_reduction": (preds == 0).sum() / len(preds),
            "accuracy": accuracy_score(truths, preds),
        })
    return pd.DataFrame(metrics)


def summarize_overall_metrics(all_decisions: pd.DataFrame) -> dict:
    preds = all_decisions["decision"].astype(int)
    truths = all_decisions["label_included"].astype(int)
    return {
        "precision": precision_score(truths, preds, zero_division=0),
        "recall": recall_score(truths, preds, zero_division=0),
        "accuracy": accuracy_score(truths, preds),
        "workload_reduction": (preds == 0).sum() / len(preds),
    }


def dataset_summary_stats(metrics_df: pd.DataFrame) -> dict:
    if metrics_df.empty:
        return {
            "median_recall": np.nan,
            "average_recall": np.nan,
            "median_workload_reduction": np.nan,
            "average_workload_reduction": np.nan,
        }
    return {
        "median_recall": float(metrics_df["recall"].median()),
        "average_recall": float(metrics_df["recall"].mean()),
        "median_workload_reduction": float(metrics_df["workload_reduction"].median()),
        "average_workload_reduction": float(metrics_df["workload_reduction"].mean()),
    }


def persist_metrics(all_decisions: pd.DataFrame) -> None:
    if all_decisions.empty:
        print("No decision logs found. Skipping metric aggregation.")
        return

    dataset_metrics = compute_dataset_metrics(all_decisions)
    dataset_metrics_path = Path(OUTPUT_DIR) / "dataset_metrics.csv"
    dataset_metrics.to_csv(dataset_metrics_path, index=False)
    print(f"Saved per-dataset metrics to {dataset_metrics_path}")

    overall_stats = summarize_overall_metrics(all_decisions)
    unfiltered_summary = dataset_summary_stats(dataset_metrics)

    filtered_decisions = all_decisions[
        ~all_decisions["dataset"].isin(EXCLUDED_DATASETS)
    ]
    filtered_metrics = dataset_metrics[
        ~dataset_metrics["dataset"].isin(EXCLUDED_DATASETS)
    ]
    filtered_overall = summarize_overall_metrics(filtered_decisions) if not filtered_decisions.empty else {
        "precision": np.nan,
        "recall": np.nan,
        "accuracy": np.nan,
        "workload_reduction": np.nan,
    }
    filtered_summary = dataset_summary_stats(filtered_metrics)

    overall_metrics = pd.DataFrame(
        [
            {
                "scope": "all_datasets",
                **overall_stats,
                **unfiltered_summary,
            },
            {
                "scope": "filtered_datasets",
                **filtered_overall,
                **filtered_summary,
            },
        ]
    )
    overall_metrics_path = Path(OUTPUT_DIR) / "overall_metrics.csv"
    overall_metrics.to_csv(overall_metrics_path, index=False)
    print(f"Saved aggregated metrics to {overall_metrics_path}")

    print(
        f"Overall metrics (all datasets): precision={overall_stats['precision']:.3f}, recall={overall_stats['recall']:.3f}, workload_reduction={overall_stats['workload_reduction']:.3f}"
    )
    print(
        f"  Dataset-level recall — median={unfiltered_summary['median_recall']:.3f}, average={unfiltered_summary['average_recall']:.3f}; workload_reduction median={unfiltered_summary['median_workload_reduction']:.3f}, average={unfiltered_summary['average_workload_reduction']:.3f}"
    )
    if not filtered_decisions.empty:
        print(
            f"Filtered metrics (excluding flawed studies): precision={filtered_overall['precision']:.3f}, recall={filtered_overall['recall']:.3f}, workload_reduction={filtered_overall['workload_reduction']:.3f}"
        )
        print(
            f"  Dataset-level recall — median={filtered_summary['median_recall']:.3f}, average={filtered_summary['average_recall']:.3f}; workload_reduction median={filtered_summary['median_workload_reduction']:.3f}, average={filtered_summary['average_workload_reduction']:.3f}"
        )
    else:
        print("Filtered metrics unavailable because no non-flagged datasets have been processed yet.")

    false_negatives = all_decisions[
        (all_decisions["label_included"] == 1) & (all_decisions["decision"] == 0)
    ]
    false_negatives_path = Path(OUTPUT_DIR) / "false_negatives.csv"
    false_negatives.to_csv(false_negatives_path, index=False)
    print(f"False negatives saved to {false_negatives_path}")

    if not false_negatives.empty:
        for dataset_key, group in false_negatives.groupby("dataset"):
            reasons = group["decision_reason"].unique()
            print(
                f"{dataset_key}: {len(group)} false negatives (reasons: {', '.join(reasons)})"
            )


@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv(str(ENV_FILE))],
    volumes={"/data/output": results_volume},
    timeout=86399,
    cpu=2,
)
async def run_experiment():
    from openai import AsyncOpenAI

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    llm_client = AsyncOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1/",
    )

    system_prompt = """
 You are an expert systematic reviewer tasked with determinng whether,
    a given set of criteria are met by a given abstract and title.
    You are generally to favor higher recall (sensitivity), as it is
    far better to have false positives (include an abstract that doesn't actually meet the criteria)
    than to have false negatives (exclude an abstract that does meet the criteria).
    It is important to remember that you cannot exclude an abstract merely because of
    missing information. For example, information on study design, populations, sampling methods,
    and many other pieaces of information might be missing. Therefore, do not
    always exclude.

    The following is EXTREMELY important. You are only to return a single integer value,
    0, 1 or 2. They have the following meanings:
    0: The abstract does not meet the criteria
    1: Unsure whether the abstract meets the criteria
    2: The abstract does meet the criteria
"""

    experiment = LLMAbstractScreeningExperiment(llm_client)

    dataset_files = sorted(
        f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")
    )
    processed_keys = {
        path.name.replace("_decisions.csv", "")
        for path in Path(OUTPUT_DIR).glob("*_decisions.csv")
    }

    for data_file in dataset_files:
        key = Path(data_file).stem
        if key in processed_keys:
            print(f"Skipping {key} because it was already processed")
            continue

        print("Processing", key)
        try:
            summary, log = await experiment.run_experiment(key, system_prompt)
            decision_path = decision_log_path(key)
            log.to_csv(decision_path, index=False)
            print("Saved log for", key)
            print("Summary:", summary)
        except Exception as exc:
            print(f"Failed on {key}: {exc}")

    all_logs = gather_decision_logs()
    persist_metrics(all_logs)
