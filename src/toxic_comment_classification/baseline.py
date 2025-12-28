from __future__ import annotations

import mlflow
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from .data import download_raw_dataset, load_from_disk_raw
from .utils import get_git_commit_id


def run_baseline(cfg: DictConfig) -> None:
    download_raw_dataset(cfg.data.paths.raw_dir, cfg.data.dataset.hf_name)
    ds = load_from_disk_raw(cfg.data.paths.raw_dir)
    df = ds.to_pandas()

    text_col = str(cfg.data.dataset.text_col)
    label_cols = list(cfg.data.dataset.label_cols)

    texts = df[text_col].astype(str).values
    y = df[label_cols].astype(int).values
    has_any = (y.sum(axis=1) > 0).astype(int)

    seed = int(cfg.data.split.seed)
    train_size = float(cfg.data.split.train_size)
    val_size = float(cfg.data.split.val_size)
    test_size = float(cfg.data.split.test_size)
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    x_train, x_tmp, y_train, y_tmp, _s_train, s_tmp = train_test_split(
        texts,
        y,
        has_any,
        test_size=(1.0 - train_size),
        random_state=seed,
        stratify=has_any,
    )

    rel_val = val_size / (val_size + test_size)
    _x_val, x_test, _y_val, y_test = train_test_split(
        x_tmp,
        y_tmp,
        test_size=(1.0 - rel_val),
        random_state=seed,
        stratify=s_tmp,
    )

    thr = float(cfg.train.threshold) if "train" in cfg and "threshold" in cfg.train else 0.5

    mlflow.set_tracking_uri(str(cfg.logging.mlflow.tracking_uri))
    mlflow.set_experiment(str(cfg.logging.mlflow.experiment_name))

    with mlflow.start_run(run_name=str(cfg.logging.mlflow.run_name)) as run:
        print("MLflow run_id:", run.info.run_id)

        mlflow.set_tag("git_commit", get_git_commit_id())
        mlflow.log_param("baseline_model", "tfidf(1-2gram)+ovr_logreg")
        mlflow.log_param("seed", seed)
        mlflow.log_param("threshold", thr)

        vec = TfidfVectorizer(
            lowercase=bool(cfg.data.text.lowercase),
            ngram_range=(1, 2),
            max_features=200000,
            min_df=2,
            strip_accents="unicode",
        )

        x_train_vec = vec.fit_transform(x_train)
        x_test_vec = vec.transform(x_test)

        x_train_vec = x_train_vec.tocsr().copy()
        x_test_vec = x_test_vec.tocsr().copy()

        base_lr = LogisticRegression(solver="liblinear", max_iter=200)
        clf = OneVsRestClassifier(base_lr, n_jobs=1)
        clf.fit(x_train_vec, y_train)

        probs = clf.predict_proba(x_test_vec)
        pred = (probs >= thr).astype(int)

        roc_auc_macro = roc_auc_score(y_test, probs, average="macro")
        f1_micro = f1_score(y_test, pred, average="micro", zero_division=0)
        f1_macro = f1_score(y_test, pred, average="macro", zero_division=0)

        print("Baseline metrics:")
        print("  test/roc_auc_macro =", float(roc_auc_macro))
        print("  test/f1_micro      =", float(f1_micro))
        print("  test/f1_macro      =", float(f1_macro))

        mlflow.log_metric("test/roc_auc_macro", float(roc_auc_macro))
        mlflow.log_metric("test/f1_micro", float(f1_micro))
        mlflow.log_metric("test/f1_macro", float(f1_macro))

        for i, label in enumerate(label_cols):
            try:
                auc_i = roc_auc_score(y_test[:, i], probs[:, i])
                mlflow.log_metric(f"test/auc_{label}", float(auc_i))
            except ValueError:
                pass
