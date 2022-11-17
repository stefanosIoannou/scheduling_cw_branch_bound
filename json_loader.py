
import json
import numpy as np

processing_times = {
    'onnx': 1,
    'emboss': 1,
    'vii': 1,
    'muse': 1,
    'night': 1,
    'blur': 1,
    'wave': 1
}

def load_json(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data

def get_data(file_name):
    data = load_json(file_name)
    due_dates = data['workflow_0']['due_dates']
    keys = list(due_dates.keys())

    p = []
    for key in keys:
        p.append(processing_times[key.split('_')[0]])

    d = []
    for key in keys:
        d.append(due_dates[key])

    no_of_jobs = len(keys)

    G = np.zeros((no_of_jobs, no_of_jobs))
    edge_set = data['workflow_0']['edge_set']
    for edge in edge_set:
        index_0 = keys.index(edge[0])
        index_1 = keys.index(edge[1])
        G[index_0, index_1] = 1
    J = list(zip(range(len(p)),p,d))
    return J, G, p, d, no_of_jobs

get_data('input.json')