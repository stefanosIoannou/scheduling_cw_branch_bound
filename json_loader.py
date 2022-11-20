
import json
import numpy as np

processing_times = {
    'vii': 19.2204,
    'blur': 5.5903,
    'night': 22.7764,
    'onnx': 2.9165,
    'emboss': 1.7308,
    'muse': 14.7329,
    'wave': 10.6599,
}

def load_json(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data

def get_precessing_times(keys):
    p = []
    for key in keys:
        p.append(processing_times[key.split('_')[0]])
    return p

def get_data(file_name):
    data = load_json(file_name)
    due_dates = data['workflow_0']['due_dates']
    keys = list(due_dates.keys())

    p = get_precessing_times(keys)

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

# get_data('input.json')
print(get_precessing_times(['onnx_1', 'muse_1', 'emboss_1', 'emboss_2', 'blur_1', 'emboss_3', 'vii_1', 'blur_2', 'wave_1', 'blur_3', 'blur_4', 'emboss_4', 'onnx_2', 'onnx_3', 'blur_5', 'wave_2', 'wave_3', 'wave_4', 'emboss_5', 'onnx_4', 'emboss_6', 'onnx_5', 'vii_2', 'blur_6', 'night_1', 'muse_2', 'emboss_7', 'onnx_6', 'wave_5', 'emboss_8', 'muse_3']))