
from queue import PriorityQueue
import numpy as np
from task import *


q = PriorityQueue()
# entry structure: (tardiness, [list of jobs], set(jobs already scheduled))



def calc_tardiness_i(list_of_job_indexes, reverse=False):
    tardiness = 0
    end_time = total_processing_time
    if reverse:
        for j in reversed(list_of_job_indexes):
            job = J[j]
            tardiness += max(0, end_time - job[2])
            end_time -= job[1]
    else:
        for j in list_of_job_indexes:
            job = J[j]
            tardiness += max(0, end_time - job[2])
            end_time -= job[1]
    return tardiness




total_processing_time = sum(p)


# Add the possible initial jobs to the priority queue
for j, processing_time, due_date in J:
    # print('This is a job', j)
    # print(tardiness)
    # Bit Magic
    # bin_set = 1
    # bin_set &= ~(1 << j)

    # Add jobs that are not prerequisites for other jobs
    dependencies = np.where(G[j] == 1)[0]
    if len(dependencies) == 0:
        tardiness = calc_tardiness_i([j])
        q.put((tardiness, [j], {j}))

def get_best_schedule():
    i = 0
    while not q.empty():
        print(i)
        i += 1

        # Get the front of the queue
        tardiness, list_of_jobs, set_of_jobs = q.get()
        # If all jobs have been allocated (and it is the minimum), we have found the best schedule
        if len(list_of_jobs) == no_of_jobs:
            return [j_i for j_i in reversed(list_of_jobs)]


        # for job in range(no_of_jobs):
        counter = 4
        for job in np.argsort(d):
            if counter <= 0:
                break
            # Working in reversed order

            # If job has already been scheduled, then skip
            if job in set_of_jobs:
                continue

            # For a job to be scheduled, the jobs that depend on it need to already been scheduled, otherwise it will
            # schedule jobs that lead to a sequence that does not respect precedence
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
            new_tardiness = new_tardiness + 0.075*new_tardiness/len(local_joblist)*(no_of_jobs-len(local_joblist))
            local_jobset.add(job)
            q.put((new_tardiness, local_joblist, local_jobset))
            counter -= 1

print(J)
schedule = get_best_schedule()
print(schedule)
print(calc_tardiness_i(schedule, True))
jan = [30, 4, 10, 3, 23, 14, 20, 22, 21, 19, 18, 9, 8, 7, 6, 17, 16, 29, 28, 27, 26, 25, 24, 15, 13, 12, 11, 5, 2, 1, 31]
jan = [j_i - 1 for j_i in jan]

our = [29, 3, 2, 22, 21, 20, 19, 18, 17, 13, 9, 8, 7, 6, 5, 16, 15, 28, 27, 26, 25, 24, 23, 14, 12, 11, 10, 4, 1, 0, 30]

# print(calc_tardiness_i(jan))

print(calc_tardiness_i(our, True))
print(calc_tardiness_i(jan, True))
