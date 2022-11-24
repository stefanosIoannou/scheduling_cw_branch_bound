import numpy as np

no_of_jobs = 31

## Node Types (Appendix A)
node_types = dict()
node_types[1] = 'onnx_1'
node_types[2] = 'muse_1'
node_types[3] = 'emboss_1'
node_types[4] = 'emboss_2'
node_types[5] = 'blur_1'
node_types[6] = 'emboss_3'
node_types[7] = 'vii_1'
node_types[8] = 'blur_2'
node_types[9] = 'wave_1'
node_types[10] = 'blur_3'
node_types[11] = 'blur_4'
node_types[12] = 'emboss_4'
node_types[13] = 'onnx_2'
node_types[14] = 'onnx_3'
node_types[15] = 'blur_5'
node_types[16] = 'wave_2'
node_types[17] = 'wave_3'
node_types[18] = 'wave_4'
node_types[19] = 'emboss_5'
node_types[20] = 'onnx_4'
node_types[21] = 'emboss_6'
node_types[22] = 'onnx_5'
node_types[23] = 'vii_2'
node_types[24] = 'blur_6'
node_types[25] = 'night_1'
node_types[26] = 'muse_2'
node_types[27] = 'emboss_7'
node_types[28] = 'onnx_6'
node_types[29] = 'wave_5'
node_types[30] = 'emboss_8'
node_types[31] = 'muse_3'


## DAG (Workflow)
G = np.zeros((no_of_jobs, no_of_jobs))
G[0, 30] = 1
G[1, 0] = 1
G[2, 7] = 1
G[3, 2] = 1
G[4, 1] = 1
G[5, 15] = 1
G[6, 5] = 1
G[7, 6] = 1
G[8, 7] = 1
G[9, 8] = 1
G[10, 4] = 1
G[11, 4] = 1
G[12, 11] = 1
G[13, 12] = 1
G[14, 10] = 1
G[15, 14] = 1
G[16, 15] = 1
G[17, 16] = 1
G[18, 17] = 1
G[19, 18] = 1
G[20, 17] = 1
G[21, 20] = 1
G[22, 21] = 1
G[23, 4] = 1
G[24, 23] = 1
G[25, 24] = 1
G[26, 25] = 1
G[27, 25] = 1
G[28, 27] = 1
G[29, 3] = 1
G[29, 9] = 1
G[29, 13] = 1
G[29, 19] = 1
G[29, 22] = 1
G[29, 26] = 1
G[29, 28] = 1


def get_tuple_list_q1():
    """
    Return a list of tuples. Each tuple corresponds to a process and follows the following signature:
    (index_of_job, processing_time, due_date). This method uses the processing times acquired from Q1,
    and is dedicated to Q2.
    """
    d = [172, 82, 18, 61, 93, 71, 217, 295, 290, 287, 253, 307, 279, 73, 355, 34, 233, 77, 88, 122, 71, 181, 340, 141,
         209, 217, 256, 144, 307, 329, 269]

    processing_times = dict()
    processing_times['vii'] = 18.8566
    processing_times['blur'] = 5.7161
    processing_times['night'] = 22.5914
    processing_times['onnx'] = 2.9634
    processing_times['emboss'] = 1.7741
    processing_times['muse'] = 14.6653
    processing_times['wave'] = 11.0234

    p = []

    for i in range(no_of_jobs):
        p.append(processing_times[node_types[i + 1].split('_')[0]])

    return list(zip(range(len(p)), p, d)), p, d

def get_tuple_list_q3():
    """
    Return a list of tuples. Each tuple corresponds to a process and follows the following signature:
    (index_of_job, processing_time, due_date). This method uses the processing times for Q3,
    """
    d = [172, 82, 18, 61, 93, 71, 217, 295, 290, 287, 253, 307, 279, 73, 355, 34, 233, 77, 88, 122, 71, 181, 340, 141,
         209, 217, 256, 144, 307, 329, 269]

    p = [4, 17, 2, 2, 6, 2, 21, 6, 13, 6, 6, 2, 4, 4, 6, 13, 13, 13, 2, 4, 2, 4, 21, 6, 25, 17, 2, 4, 13, 2, 17]

    return list(zip(range(len(p)), p, d)), p, d