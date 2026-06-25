"""Load simulation states and metadata."""

import json
from pathlib import Path

import numpy as np

from utils.saving import INIT_STATES_DIR, SEEDS_DIR


def load_initial_states(load_path):
    load_path = Path(load_path) / INIT_STATES_DIR
    x = np.load(load_path / "x.npy")
    y = np.load(load_path / "y.npy")
    t = np.load(load_path / "t.npy")
    return x, y, t


def load_sweep(load_path):
    with open(Path(load_path) / "sweep.json", "r") as fp:
        return json.load(fp)


def load_flags(load_path):
    with open(Path(load_path) / "flags.json", "r") as fp:
        return json.load(fp)


def load_seeds(load_path):
    load_path = Path(load_path) / SEEDS_DIR
    return np.load(load_path / "seeds.npy")
