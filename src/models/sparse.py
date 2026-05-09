from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

def _normalize_classifier(name: str) -> str:
    n = str(name).strip().lower()
    if n in ("linsvc", "linear_svc", "linearsvc"):
        return "linear_svc"
    if n in ("logreg", "logistic", "lr"):
        return "logreg"
    if n == "sgd":
        return "sgd"
    raise ValueError(f"Unsupported classifier: {name}")


def _build_vectorizer(model_cfg: dict[str, Any]) -> FeatureUnion:
    max_f = int(model_cfg.get("max_features", 200_000))
    char_max = max(1, max_f // 2)
    word_max = max(1, max_f - char_max)
    n_w = model_cfg.get("ngram_word", [1, 2])
    n_c = model_cfg.get("ngram_char", [3, 5])
    word_range = (int(n_w[0]), int(n_w[1])) if len(n_w) >= 2 else (1, 2)
    char_range = (int(n_c[0]), int(n_c[1])) if len(n_c) >= 2 else (3, 5)
    min_df = int(model_cfg.get("min_df", 2))
    max_df = float(model_cfg.get("max_df", 1.0))
    sub = bool(model_cfg.get("sublinear_tf", True))

    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_range,
        min_df=min_df,
        max_df=max_df,
        max_features=char_max,
        sublinear_tf=sub,
        dtype=np.float64,
    )
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_range,
        min_df=min_df,
        max_df=max_df,
        max_features=word_max,
        sublinear_tf=sub,
        dtype=np.float64,
    )
    return FeatureUnion([("char", char), ("word", word)])


def _build_classifier(
    name: str,
    model_cfg: dict[str, Any],
    n_classes: int,
    random_state: int,
):
    c = float(model_cfg.get("C", 1.0))
    max_iter = int(model_cfg.get("max_iter", 5000))
    cw = model_cfg.get("class_weight", None)
    n_jobs = int(model_cfg.get("n_jobs", 1))
    n_classes_eff = n_classes
    if name == "linear_svc":
        return LinearSVC(
            C=c,
            class_weight=cw,
            max_iter=max_iter,
            random_state=random_state,
            dual=False,
        )
    if name == "logreg":
        return LogisticRegression(
            C=c,
            class_weight=cw,
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            solver="saga" if n_classes_eff > 2 else "lbfgs",
        )
    if name == "sgd":
        return SGDClassifier(
            loss="log_loss" if n_classes_eff > 2 else "hinge",
            alpha=1.0 / (c * max(1, n_classes_eff)) if c > 0 else 1e-4,
            max_iter=max_iter,
            class_weight=cw,
            random_state=random_state,
            n_jobs=n_jobs,
            tol=1e-3,
        )
    raise ValueError(f"Unknown classifier: {name}")


def fit(
    train_df: pd.DataFrame,
    config: dict[str, Any],
    text_col: str = "text",
    topic_col: str = "topic_id",
) -> dict[str, Any]:
    """Fit TF-IDF (char + word) + linear classifier. Classes = sorted unique train ``topic_id``."""
    if text_col not in train_df.columns or topic_col not in train_df.columns:
        raise KeyError("train_df must include text and topic_id columns")
    model_cfg = config.get("model", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    seed = int(config.get("seed", 42))
    clf_name = _normalize_classifier(str(model_cfg.get("classifier", "linear_svc")))

    if train_df[topic_col].isna().any():
        raise ValueError("train_df must not contain null topic_id values")
    y_str = train_df[topic_col].astype(str)
    classes = sorted(set(y_str.tolist()))
    if not classes:
        raise ValueError("No class labels in training data")
    label_to_i = {lab: j for j, lab in enumerate(classes)}

    texts = train_df[text_col].astype(str).tolist()
    y = np.array([label_to_i[s] for s in y_str.tolist()])

    vectorizer = _build_vectorizer(model_cfg)
    clf = _build_classifier(clf_name, model_cfg, n_classes=len(classes), random_state=seed)

    pipe = Pipeline(
        [
            ("vect", vectorizer),
            ("clf", clf),
        ]
    )
    pipe.fit(texts, y)
    return {
        "pipeline": pipe,
        "vectorizer": vectorizer,
        "classifier": clf,
        "classes": classes,
        "classifier_name": clf_name,
        "config": config,
    }


def predict_topk(
    model: dict[str, Any],
    texts: list[str],
    k: int = 10,
) -> tuple[list[list[str]], list[list[float]], list[float]]:
    """
    Return (topic_ids, scores, latency_ms) per item.

    For ``linear_svc`` / ``hinge``-SGD: higher ``decision_function`` is better;
    for ``logreg`` / log-loss SGD: uses ``predict_proba``.
    """
    pipe: Pipeline = model["pipeline"]
    classes: list[str] = list(model["classes"])
    clf = pipe.named_steps["clf"]
    clf_name = str(model.get("classifier_name", "linear_svc"))

    top_ids: list[list[str]] = []
    top_scores: list[list[float]] = []
    latencies: list[float] = []
    n_classes = len(classes)
    k_eff = min(k, n_classes) if n_classes else 0

    use_proba = clf_name == "logreg"

    for text in texts:
        t0 = time.perf_counter()
        if use_proba:
            proba = pipe.predict_proba([text])
            p = proba[0]
            if k_eff <= 0:
                top_ids.append([])
                top_scores.append([])
            else:
                idx = np.argsort(-p)[:k_eff]
                top_ids.append([classes[i] for i in idx])
                top_scores.append([float(p[i]) for i in idx])
        else:
            dfun = pipe.decision_function([text])
            row = dfun[0] if dfun.ndim == 1 else dfun[0]
            if k_eff <= 0:
                top_ids.append([])
                top_scores.append([])
            else:
                idx = np.argsort(-row)[:k_eff]
                top_ids.append([classes[i] for i in idx])
                top_scores.append([float(row[i]) for i in idx])
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return top_ids, top_scores, latencies


def save(model: dict[str, Any], out_dir: str | Path) -> None:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    pipeline = model["pipeline"]
    joblib.dump(pipeline, target / "model.pkl")
    # duplicate vectorizer for contract (also extractable from pipeline)
    v = pipeline.named_steps.get("vect")
    joblib.dump(v, target / "vectorizer.pkl")
    with (target / "classes.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(list(model["classes"]), stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def load(artifacts_dir: str | Path) -> dict[str, Any]:
    p = Path(artifacts_dir)
    pipeline: Pipeline = joblib.load(p / "model.pkl")
    with (p / "classes.json").open("r", encoding="utf-8") as stream:
        classes: list[str] = json.load(stream)
    clf = pipeline.named_steps["clf"]
    if isinstance(clf, LogisticRegression):
        clf_name = "logreg"
    elif isinstance(clf, SGDClassifier):
        clf_name = "sgd"
    else:
        clf_name = "linear_svc"
    return {
        "pipeline": pipeline,
        "vectorizer": pipeline.named_steps.get("vect"),
        "classifier": clf,
        "classes": classes,
        "classifier_name": clf_name,
        "config": {},
    }