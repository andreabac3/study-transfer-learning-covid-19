import os
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import yaml

warnings.filterwarnings("ignore")

import torch
import hydra
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from src.callbacks import ImagePredictionLogger
from src.common.utils import load_envs, PROJECT_ROOT, set_determinism_the_old_way, gpus, enable_16precision
from src.pl_data_modules import BasePLDataModule
from src.pl_modules import BasePLModule

# Load environment variables
load_envs()



def build_callbacks(cfg: omegaconf.DictConfig, checkpoint_path: str = "") -> List[Callback]:
    # callbacks declaration
    callbacks_store: List[Callback] = [RichProgressBar()]

    if cfg.train.apply_early_stopping:
        callbacks_store.append(EarlyStopping(**cfg.train.early_stopping))

    callbacks_store.append(
        ModelCheckpoint(
            **cfg.train.model_checkpoint,
            dirpath=checkpoint_path,
        )
    )
    return callbacks_store


def get_dataset_info(conf: omegaconf.DictConfig) -> Dict[str, List[Dict[str, int]]]:
    """
    Return dict which contains each split (train, validation, test) with the number of samples belonging to the labels selected
    e.g.
     {'train': [{'COVID': 563}, {'bacteria': 1285}, {'virus': 1035}, {'normal': 950}],
     'validation': [{'COVID': 15}, {'bacteria': 16}, {'virus': 16}, {'normal': 16}],
     'test': [{'COVID': 17}, {'bacteria': 18}, {'virus': 18}, {'normal': 18}]}
    """
    class_to_use: List[str] = conf.labels.class_to_use
    dataset_path: Dict[str, str] = {
        "train": conf.data.dataset.train_path,
        "validation": conf.data.dataset.validation_path,
        "test": conf.data.dataset.test_path,
    }
    dataset_info: Dict[str, List[Dict[str, int]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for key, path in dataset_path.items():
        for label in class_to_use:
            num_samples = len(os.listdir(f"{path}/{label}"))
            label_num_samples: Dict[str, int] = {label: num_samples}
            dataset_info[key].append(label_num_samples)
    return dataset_info


def train(conf: omegaconf.DictConfig) -> None:
    # reproducibility
    pl.seed_everything(conf.train.seed)
    set_determinism_the_old_way(True)
    logger = None


    if conf.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{conf.train.pl_trainer.fast_dev_run}>. Forcing debugger configuration"
        )
        # Debuggers don't like GPUs nor multiprocessing
        conf.train.pl_trainer.gpus = 0
        conf.train.pl_trainer.precision = 32
        conf.data.datamodule.num_workers = {k: 0 for k in conf.data.datamodule.num_workers}
        # Switch wandb mode to offline to prevent online logging
        conf.logging.wandb.mode = "offline"

    # data module declaration
    pl_data_module = BasePLDataModule(conf)

    # main module declaration
    pl_module = BasePLModule(conf)

    # callbacks declaration

    if conf.logging.log:
        class_to_use_str: str = (
            str(conf.labels.class_to_use).strip("[]").replace(",", "-").replace("'", "").replace(" ", "")
        )
        hydra.utils.log.info(f"Instantiating <WandbLogger>")
        wandb_config = conf.logging
        logger = WandbLogger(
            **conf.logging.wandb,
            group=f"{conf.data.dataset.name}-{class_to_use_str}-{conf.data.subset_percentage}",
            name=conf.train.model_type,
            log_model=True,
        )
        hydra.utils.log.info(f"W&B is now watching <{wandb_config.watch.log}>!")
        logger.watch(pl_module, log=wandb_config.watch.log, log_freq=wandb_config.watch.log_freq)

        dataset_info = get_dataset_info(conf=conf)
        with open(Path(logger.experiment.dir) / "dataset-info.yaml", "w") as outfile:
            yaml.dump(dataset_info, outfile, default_flow_style=False)
        yaml_conf: str = OmegaConf.to_yaml(cfg=conf)
        (Path(logger.experiment.dir) / "hparams.yaml").write_text(yaml_conf)

    ckpt_folder: str = hydra.utils.to_absolute_path(f"./ckpt/{conf.data.subset_percentage}/{conf.train.model_type}/")
    os.makedirs(ckpt_folder, exist_ok=True)

    callbacks_store = [RichProgressBar()] if conf.train.pl_trainer.fast_dev_run else build_callbacks(cfg=conf, checkpoint_path=ckpt_folder)
    
    if not torch.cuda.is_available():
        conf.train.pl_trainer.gpus = 0
        conf.train.pl_trainer.precision = 32

    trainer = pl.Trainer(
        **conf.train.pl_trainer,
        callbacks=callbacks_store,
        logger=logger,
    )

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)
    if conf.logging.log:
        logger.experiment.finish()


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    import os

    print("Working directory : {}".format(os.getcwd()))
    train(conf)


if __name__ == "__main__":
    main()
