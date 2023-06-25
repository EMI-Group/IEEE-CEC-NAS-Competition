import json
import os
import shutil
import hashlib
import numpy as np
import pandas as pd
from pymoo.util.ref_dirs import get_reference_directions


def folder_create(path):
    if os.path.isfile(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.mkdir(path)
    return path


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Record:
    def __init__(self, run, F, HV, IGD=None):
        self.run = run
        self.F = F
        self.IGD = IGD
        self.HV = HV

    @staticmethod
    def to_pd(records):
        return pd.DataFrame(
            {'run': i.run,
             'IGD': i.IGD,
             'HV': i.HV
             } for i in records
        )

def get_benchmark_settings(n_obj):
    n_gen = 100

    if n_obj == 2:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=99)
    elif n_obj == 3:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=13)
    elif n_obj == 4:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=7)
    elif n_obj == 5:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=5)
    elif n_obj == 6:
        ref_dirs = get_reference_directions(
            "multi-layer",
            get_reference_directions("das-dennis", n_obj, n_partitions=4, scaling=1.0),
            get_reference_directions("das-dennis", n_obj, n_partitions=1, scaling=0.5))
    elif n_obj == 8:
        ref_dirs = get_reference_directions(
            "multi-layer",
            get_reference_directions("das-dennis", n_obj, n_partitions=3, scaling=1.0),
            get_reference_directions("das-dennis", n_obj, n_partitions=2, scaling=0.5))
    else:
        raise NotImplementedError

    pop_size = ref_dirs.shape[0]

    return pop_size, n_gen, ref_dirs


def md5_file(path):
    with open(path, 'rb') as fp:
        data = fp.read()
    return hashlib.md5(data).hexdigest()

def md5(s):
    md5hash = hashlib.md5(s)
    return md5hash.hexdigest()
