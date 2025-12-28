from __future__ import annotations

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from .data import download_raw_dataset, make_loaders, make_splits_and_vocab
from .lit_module import ToxicLitModule
from .models import TextCNNConfig
from .utils import ensure_dir, get_git_commit_id, try_dvc_pull
from .vocab import Vocab


def train_main(cfg: DictConfig) -> None:
    pl.seed_everything(int(cfg.train.seed), workers=True)
    try_dvc_pull(["data/raw/jigsaw", "data/processed"])
    download_raw_dataset(cfg.data.paths.raw_dir, cfg.data.dataset.hf_name)

    processed = make_splits_and_vocab(
        raw_dir=cfg.data.paths.raw_dir,
        processed_dir=cfg.data.paths.processed_dir,
        vocab_path=cfg.data.paths.vocab_path,
        text_col=cfg.data.dataset.text_col,
        label_cols=list(cfg.data.dataset.label_cols),
        train_size=float(cfg.data.split.train_size),
        val_size=float(cfg.data.split.val_size),
        test_size=float(cfg.data.split.test_size),
        seed=int(cfg.data.split.seed),
        lowercase=bool(cfg.data.text.lowercase),
        vocab_size=int(cfg.data.text.vocab_size),
        min_freq=int(cfg.data.text.min_freq),
    )

    vocab = Vocab.from_json(processed.vocab_path)

    train_loader, val_loader, _ = make_loaders(
        train_path=processed.train_path,
        val_path=processed.val_path,
        test_path=processed.test_path,
        vocab=vocab,
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
        max_len=int(cfg.data.text.max_len),
        lowercase=bool(cfg.data.text.lowercase),
    )

    model_cfg = TextCNNConfig(
        vocab_size=len(vocab.token_to_id),
        embed_dim=int(cfg.model.embed_dim),
        num_filters=int(cfg.model.num_filters),
        kernel_sizes=list(cfg.model.kernel_sizes),
        dropout=float(cfg.model.dropout),
        num_labels=len(cfg.data.dataset.label_cols),
    )

    lit = ToxicLitModule(
        model_cfg=model_cfg,
        pad_id=vocab.pad_id,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        threshold=float(cfg.train.threshold),
    )

    ensure_dir(cfg.train.checkpoint.dirpath)
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.train.checkpoint.dirpath,
        filename="best",
        monitor=cfg.train.checkpoint.monitor,
        mode=cfg.train.checkpoint.mode,
        save_top_k=int(cfg.train.checkpoint.save_top_k),
    )

    mlf_logger = MLFlowLogger(
        tracking_uri=cfg.logging.mlflow.tracking_uri,
        experiment_name=cfg.logging.mlflow.experiment_name,
        run_name=cfg.logging.mlflow.run_name,
    )

    commit_id = get_git_commit_id()
    mlf_logger.experiment.set_tag(mlf_logger.run_id, "git_commit", commit_id)
    mlf_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    trainer = pl.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        logger=mlf_logger,
        callbacks=[checkpoint_cb],
        precision=cfg.train.precision,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        log_every_n_steps=int(cfg.train.log_every_n_steps),
    )

    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)
