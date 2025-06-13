"""This module contains the data loading functions for the datasets used in the project."""

from .banknote import load_banknote
from .exist_syn import load_banana, load_twonorm
from .mammographic import load_mammographic
from .mushroom import load_mushroom
from .nursery import load_nursery
from .obesity import load_obesity
from .sms_spam import load_smsspam
from .synthetic import gen_blobs, gen_circles, gen_moons, gen_normal_precise_1d, gen_xor

synthetic_datasets = [
    "gen_blobs",
    "gen_circles",
    "gen_moons",
    "gen_normal_precise_1d",
    "gen_xor",
    "load_twonorm",
    "load_banana",
]

natural_datasets = [
    "load_mushroom",
    "load_mammographic",
    "load_nursery",
    "load_obesity",
    "load_banknote",
    "load_smsspam",
]

__all__ = synthetic_datasets + natural_datasets
