from queue import PriorityQueue
import numpy as np
from task import *


# entry structure: (tardiness, [list of jobs], set(jobs already scheduled))

def calc_tardiness_i(list_of_job_indexes, reverse=False):
    # Calculate the tardiness of the list of the schedule of jobs
    tardiness = 0
    # Total Processing Time
    end_time = sum(p)
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


def job_can_be_scheduled(job, set_of_jobs: set):
    # Check it the job can be schedules, i.e. the children of the job are already in the job_schedule.
    # Return true if it can be schedules, False otherwise
    dependencies = np.where(G[job] == 1)[0]
    return set_of_jobs.issubset(dependencies)


def get_best_schedule():
    q = PriorityQueue()

    # Add the possible initial jobs to the priority queue
    for j, processing_time, due_date in J:
        # Add jobs that are not prerequisites for other jobs,
        # i.e. jobs with no edges to other jobs, i.e. leaves
        dependencies = np.where(G[j] == 1)[0]
        if len(dependencies) == 0:
            tardiness = calc_tardiness_i([j])
            q.put((tardiness, [j], {j}))

    iterations = 0
    while not q.empty() and iterations < 30000:
        # print(iterations)
        iterations += 1

        # Best schedule so far, with least tardiness
        # Important! List of jobs has the jobs in reverse order: i.e. from last to first
        tardiness, list_of_jobs, set_of_jobs = q.get()

        # If all jobs have been allocated (and it is the minimum), we have found the best schedule
        if len(list_of_jobs) == no_of_jobs:
            return [j_i for j_i in reversed(list_of_jobs)]

        # Stop the loop if this many schedules have been added
        # Sort the jobs from soonest to latest
        due_dates_sorted_idxs = np.argsort(d)
        idx = 0
        # If job has already been scheduled, then skip
        while counter > 0 and idx < len(due_dates_sorted_idxs) \
                and due_dates_sorted_idxs[idx] not in set_of_jobs:
            job = due_dates_sorted_idxs[idx]
            # Working in reversed order

            if job_can_be_scheduled(job, set_of_jobs):
                local_joblist = list_of_jobs.copy()
                local_jobset = set_of_jobs.copy()
                local_joblist.append(job)
                new_tardiness = calc_tardiness_i(local_joblist)
                # new_tardiness = new_tardiness + 0.075 * new_tardiness / len(local_joblist) * (
                #         no_of_jobs - len(local_joblist))
                local_jobset.add(job)
                q.put((new_tardiness, local_joblist, local_jobset))
                counter -= 1

            idx += 1


def get_best_schedule(beam):
    q = PriorityQueue()

    # Add the possible initial jobs to the priority queue
    for j, processing_time, due_date in J:
        # Add jobs that are not prerequisites for other jobs,
        # i.e. jobs with no edges to other jobs, i.e. leaves
        dependencies = np.where(G[j] == 1)[0]
        if len(dependencies) == 0:
            tardiness = calc_tardiness_i([j])
            q.put((tardiness, [j], {j}))

    iterations = 0
    while not q.empty() and iterations < 30000:
        # print(iterations)
        iterations += 1

        # Best schedule so far, with least tardiness
        # Important! List of jobs has the jobs in reverse order: i.e. from last to first
        tardiness, list_of_jobs, set_of_jobs = q.get()

        # If all jobs have been allocated (and it is the minimum), we have found the best schedule
        if len(list_of_jobs) == no_of_jobs:
            return [j_i for j_i in reversed(list_of_jobs)]

        # Stop the loop if this many schedules have been added
        counter = beam
        # Sort the jobs from soonest to latest
        due_dates_sorted_idxs = np.argsort(d)
        idx = 0
        # If job has already been scheduled, then skip
        while counter > 0 and idx < len(due_dates_sorted_idxs) \
                and due_dates_sorted_idxs[idx] not in set_of_jobs:
            job = due_dates_sorted_idxs[idx]
            # Working in reversed order

            # For a job to be scheduled, the jobs that depend on it need to already been scheduled, otherwise it will
            # schedule jobs that lead to a sequence that does not respect precedence
            # dependencies = np.where(G[job] == 1)[0]
            # flag = True
            #
            # for dependencie in dependencies:
            #     if dependencie not in set_of_jobs:
            #         flag = False
            #         break
            # if flag is False:
            #     continue
            if job_can_be_scheduled(job, set_of_jobs):
                local_joblist = list_of_jobs.copy()
                local_jobset = set_of_jobs.copy()
                local_joblist.append(job)
                new_tardiness = calc_tardiness_i(local_joblist)
                # new_tardiness = new_tardiness + 0.075 * new_tardiness / len(local_joblist) * (
                #         no_of_jobs - len(local_joblist))
                local_jobset.add(job)
                q.put((new_tardiness, local_joblist, local_jobset))
                counter -= 1

            idx += 1


print(J)
schedule = get_best_schedule()
print(schedule)
print(calc_tardiness_i(schedule, True))

# jan = [30, 4, 10, 3, 23, 14, 20, 22, 21, 19, 18, 9, 8, 7, 6, 17, 16, 29, 28, 27, 26, 25, 24, 15, 13, 12, 11, 5, 2, 1, 31]
# jan = [j_i - 1 for j_i in jan]

# our = [29, 3, 2, 22, 21, 20, 19, 18, 17, 13, 9, 8, 7, 6, 5, 16, 15, 28, 27, 26, 25, 24, 23, 14, 12, 11, 10, 4, 1, 0, 30]

# print(calc_tardiness_i(jan))

# print(calc_tardiness_i(our, True))
# print(calc_tardiness_i(jan, True))
