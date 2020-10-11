import os
import math

import json
import numpy as np


def generate_instance(num_vertices):
    # Posição do vertex no plano [0, 1]
    points = np.random.uniform(size=(num_vertices, 2))
    # Compute the distance btw the points
    dist = {str((i, j)):
        math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
        for i in range(num_vertices) for j in range(i)}
    return [num_vertices, points, dist]


def instance2json(ins):
    names = ['num_vertices', 'points', 'dist']
    dic_ins = {}
    for name, value in zip(names, ins):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        dic_ins[name] = value
    return json.dumps(dic_ins)


def save_json(filename, str_json):
    with open(filename, 'w') as file:
        file.write(str_json)


def generate_and_save_all(save_folder='data'):
    os.makedirs(save_folder, exist_ok=True)

    # Quantidade de vértices
    V = [100, 150, 200, 250, 300]

    for index, v in enumerate(V):
       ins = generate_instance(v)
       json_ins = instance2json(ins)
       save_json(os.path.join(save_folder, f'instancia-{index}.json'), json_ins)


if __name__ == "__main__":
    generate_and_save_all()