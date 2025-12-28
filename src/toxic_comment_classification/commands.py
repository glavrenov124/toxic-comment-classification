from __future__ import annotations

import hydra
from omegaconf import DictConfig

from .baseline import run_baseline
from .train import train_main


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    command = str(cfg.get("command", "textcnn")).lower()

    if command == "textcnn":
        train_main(cfg)
        return

    if command == "baseline":
        run_baseline(cfg)
        return


if __name__ == "__main__":
    main()
