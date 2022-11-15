
from queue import PriorityQueue
import numpy as np


q = PriorityQueue()
# (tardiness, [list of jobs], set())
p = [4, 17, 2, 2, 6, 2, 21, 6, 13, 6, 6, 2, 4, 4, 6, 13, 13, 13, 2, 4, 2, 4, 21, 6, 25, 17, 2, 4, 13, 2, 17]
d = [172, 82, 18, 61, 93, 71, 217, 295, 290, 287, 253, 307, 279, 73, 355, 34, 233, 77, 88, 122, 71, 181, 340, 141, 209, 217, 256, 144, 307, 329, 269]
J = list(zip(range(len(p)),p,d))
no_of_jobs = len(p)
print('Jobs All', J)

# Matrix Of Dependencies
G = np.zeros((no_of_jobs, no_of_jobs))
G[0,30]=1
G[1,0]=1
G[2,7]=1
G[3,2]=1
G[4,1]=1
G[5,15]=1
G[6,5]=1
G[7,6]=1
G[8,7]=1
G[9,8]=1
G[10,4]=1
G[11,4]=1
G[12,11]=1
G[13,12]=1
G[14,10]=1
G[15,14]=1
G[16,15]=1
G[17,16]=1
G[18,17]=1
G[19,18]=1
G[20,17]=1
G[21,20]=1
G[22,21]=1
G[23,4]=1
G[24,23]=1
G[25,24]=1
G[26,25]=1
G[27,25]=1
G[28,27]=1
G[29,3]=1
G[29,9]=1
G[29,13]=1
G[29,19]=1
G[29,22]=1
G[29,26]=1
G[29,28]=1


def calc_tardiness_i(list_of_job_indexes):
    tardiness = 0
    end_time = total_processing_time
    for j in list_of_job_indexes:
        job = J[j]
        tardiness += max(0, end_time - job[2])
        end_time -= job[1]
    return tardiness

def calc_tardiness(jobs):
    # List of Jobs= [list of (job_index, processing_time, due_date)]
    tardiness = 0
    end_time = total_processing_time
    for job in jobs:
        tardiness += max(0, end_time - job[2])
        end_time -= job[1]
    return tardiness


# Typical Pattern
# (priority_number, data)

# Sorting
# sorted(list(entries))[0])



total_processing_time = sum(p)

# Contains the job index, processing time, and due time
#

for j, processing_time, due_date in J:
    # print('This is a job', j)
    tardiness = calc_tardiness_i([j])
    # print(tardiness)
    # Bit Magic
    # bin_set = 1
    # bin_set &= ~(1 << j)
    dependencies = np.where(G[j] == 1)[0]
    if len(dependencies) == 0:
        q.put((tardiness, [j], {j}))



def get_best_schedule():
    i = 0
    while not q.empty():
        print(i)
        i += 1
        tardiness, list_of_jobs, set_of_jobs = q.get()
        if len(list_of_jobs) == no_of_jobs:
            return [j_i for j_i in reversed(list_of_jobs)]

        for job in range(no_of_jobs):
            # Working in reversed order
            if job in set_of_jobs:
                continue
            # What  need to have already
            dependencies = np.where(G[job] == 1)[0]
            flag = True

            for dependencie in dependencies:
                if dependencie not in set_of_jobs:
                    flag = False
                    break
            if flag is False:
                continue
            local_joblist = list_of_jobs.copy()
            local_jobset = set_of_jobs.copy()
            local_joblist.append(job)
            new_tardiness = calc_tardiness_i(local_joblist)
            local_jobset.add(job)
            q.put((new_tardiness, local_joblist, local_jobset))

schedule = get_best_schedule()
print(schedule)
print(calc_tardiness_i(schedule))
jan = [30, 4, 10, 3, 23, 14, 20, 22, 21, 19, 18, 9, 8, 7, 6, 17, 16, 29, 28, 27, 26, 25, 24, 15, 13, 12, 11, 5, 2, 1, 31]
jan = [j_i - 1 for j_i in jan]

print(calc_tardiness_i(jan))
