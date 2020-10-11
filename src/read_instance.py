import os
import json
import numpy as np
from glob import glob


def list_avaliable_instances(folder):
    return glob(os.path.join(folder, "*.json"))


def process_dist(num_vertices, dist):
    res = np.zeros((num_vertices, num_vertices))
    for k, v in dist.items():
        i, j = eval(k)
        res[i, j] = res[j, i] = v
    return res


def load_instance(filename):
    json_ins = json.load(open(filename))
    return [
        json_ins["num_vertices"],
        json_ins["points"],
        {eval(k): v for k, v in json_ins["dist"].items()}
    ]
