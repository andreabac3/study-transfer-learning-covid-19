import os
import json
from pathlib import Path
from typing import Optional
from multiprocessing import cpu_count

import dotenv

import torch
from omegaconf import DictConfig



def gpus(conf: DictConfig) -> int:
    """Utility to determine the number of GPUs to use."""
    return conf.train.pl_trainer.gpus if torch.cuda.is_available() else 0


def enable_16precision(conf: DictConfig) -> int:
    """Utility to determine the number of GPUs to use."""
    return conf.train.pl_trainer.precision if torch.cuda.is_available() else 32



def set_determinism_the_old_way(deterministic: bool):
    # determinism for cudnn
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        if deterministic:
            # fixing non-deterministic part of horovod
            # https://github.com/PyTorchLightning/pytorch-lightning/pull/1572/files#r420279383
            os.environ["HOROVOD_FUSION_THRESHOLD"] = str(0)

def get_number_of_cpu_cores() -> int:
    return cpu_count()


def read_json(filename: str) -> dict:
    with open(filename, "r") as reader:
        return json.load(reader)

def write_json(filename: str, dictionary: dict) -> None:
    with open(filename, "w") as writer:
        json.dump(dictionary, writer, indent=4)


def get_env(env_name: str) -> str:
    """
    Safely read an environment variable.
    Raises errors if it is not defined or it is empty.

    :param env_name: the name of the environment variable
    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        raise KeyError(f"{env_name} not defined")
    env_value: str = os.environ[env_name]
    if not env_value:
        raise ValueError(f"{env_name} has yet to be configured")
    return env_value


def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


# Load environment variables
load_envs()

# Set the cwd to the project root
PROJECT_ROOT: Path = Path(get_env("PROJECT_ROOT"))
assert (
    PROJECT_ROOT.exists()
), "You must configure the PROJECT_ROOT environment variable in a .env file!"

os.chdir(PROJECT_ROOT)
