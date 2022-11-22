from queue import PriorityQueue
import numpy as np
from task import *
from collections import deque

# entry structure: (tardiness, [list of jobs], set(jobs already scheduled))


# def calculate_hus_heuristic():
#     # Simple breath-first search
#     # TODO: This is wrong
#     q = deque()
#     q.append((29,7))
#     # Add the root
#     hmap = {}
#     while not len(q) == 0:
#         job, dist = q.popleft()
#         hmap[job] = dist
#         for children in np.where(G[job] == 1)[0]:
#             q.append((children, dist - 1))
#     return hmap

def calc_hus_heuristic():
    hmap = dict()
    for j in range(no_of_jobs):
        hmap[j] = calc_distance_to_sink(j)
    return hmap

def get_best_schedule_w_heuristic():
    # Method without any modifications, nor iteration limitations
    iterations = 0
    q = PriorityQueue()

    # Keep the longest schedule
    longest_schedule: List = []
    longest_hue_distance = 1000000
    longest_set_of_jobs = set()
    hue_distances = calc_hus_heuristic()

    # Used for fathoming
    upper_bound = 1000000000

    # Add the possible initial jobs to the priority queue
    for j, processing_time, due_date in J:
        # Add jobs that are not prerequisites for other jobs,
        # i.e. jobs with no edges to other jobs, i.e. leaves
        dependencies = np.where(G[j] == 1)[0]
        if len(dependencies) == 0:
            q.put((hue_distances, [j], {j}))

    # max_size_of_pending_list = q.qsize()

    while not q.empty() and iterations < 30000:
        # print(iterations)
        iterations += 1
        _, list_of_jobs, set_of_jobs = q.get()
        if len(list_of_jobs) == no_of_jobs:
            return util_index2job(list(reversed(list_of_jobs)))

        # REMOVE COMMENTS TO GET CURRENT NODE AND TOTAL TARDINESS OF FIRST 2
        # AND LAST 2 ITERATIONS
        # if iterations <= 2 or iterations == 29999 or iterations == 29998:
        #     print(f'Current Node [{iterations}]:'
        #           f'{list(j + 1for j in reversed(list_of_jobs))}'
        #           f'\nTotal Tardiness:{tardiness}')

        for job in range(no_of_jobs):
            # Working in reversed order
            if job_can_be_scheduled(job, set_of_jobs):
                local_joblist = list_of_jobs.copy()
                local_jobset = set_of_jobs.copy()
                local_joblist.append(job)
                local_jobset.add(job)
                # new_tardiness = calc_tardiness_i(local_joblist)
                hue_distance = hue_distances[job]

                # Fathoming
                if hue_distance < upper_bound:
                    if len(local_joblist) == no_of_jobs:
                        if len(local_joblist) == no_of_jobs:
                            upper_bound = hue_distance
                            fathoming(upper_bound, q)

                    q.put((hue_distance, local_joblist, local_jobset))

                    # If there is a tie, this is a bit extra
                    if len(local_joblist) == len(longest_schedule):
                        # If the new tardiness is better than the old
                        if hue_distance < longest_hue_distance:
                            longest_schedule = local_joblist
                            # longest_schedule_tardiness = new_tardiness
                            longest_set_of_jobs = local_jobset
                            longest_hue_distance = hue_distance

                    elif len(local_joblist) > len(longest_schedule):
                        longest_schedule = local_joblist
                        # longest_schedule_tardiness = new_tardiness
                        longest_hue_distance = hue_distance
                        longest_set_of_jobs = local_jobset

                # max_size_of_pending_list = max(max_size_of_pending_list,q.qsize())

    # Find missing jobs
    missing_jobs = set(range(no_of_jobs)) - longest_set_of_jobs
    while len(missing_jobs) > 0:
        ready = []
        for job in missing_jobs:
            if job_can_be_scheduled(job, longest_set_of_jobs):
                ready.append(job)

        # Append the job with the highest due date, higher chances of getting a non tardy job
        job_to_w_later_due_date = max(ready, key=lambda x: d[x])
        longest_schedule.append(job_to_w_later_due_date)
        missing_jobs.remove(job_to_w_later_due_date)
        longest_set_of_jobs.add(job_to_w_later_due_date)

    # print(max_size_of_pending_list)
    return util_index2job(list(reversed(longest_schedule)))

def calc_distance_to_sink(start_j):
    stack = deque()
    stack.append((start_j,0))
    while len(stack) != 0:
        j, d = stack.pop()
        for child in np.where(G[j] == 1)[0]:
            if child == 30:
                return d+1
            else:
                stack.append((child, d+1))
    return 0

def calc_tardiness_i(list_of_job_indexes, reverse=False):
    # Calculate the tardiness of the list of the schedule of jobs.
    # Reverse is True if and only if the schedule is in the correct order
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

def calc_tardiness(list_of_job_indexes, reverse=False):
    # Calculate the tardiness of the list of the schedule of jobs.
    # Reverse is True if and only if the schedule is in the correct order
    tardiness = 0
    # Total Processing Time
    end_time = sum(p)
    if reverse:
        for j in reversed(list_of_job_indexes):
            job = J[j-1]
            tardiness += max(0, end_time - job[2])
            end_time -= job[1]
    else:
        for j in list_of_job_indexes:
            job = J[j]
            tardiness += max(0, end_time - job[2])
            end_time -= job[1]
    return tardiness


def util_is_feasible(schedule):
    # Return true if the schedule, (not reversed) is feasible.
    # This is the final schedule, jobs are not indexes!
    G_current = np.copy(G)
    for job in schedule:
        dependencies = np.where(G_current[:, job - 1] == 1)[0]
        if len(dependencies) != 0:
            for dependency in dependencies:
                print(f'{job} must be before {dependency + 1}')
            return False
        G_current[job - 1] = 0
    return True


def util_is_complete(schedule):
    # Return if the schedule is complete
    return set(range(1, 32)) == set(schedule)

def util_index2job(schedule):
    # Turn a sequence of indexes to a sequence of jobs
    return [j+1 for j in schedule]

def job_can_be_scheduled(job, set_of_jobs: set):
    # Check it the job can be schedules, i.e. the children of the job are already in the job_schedule.
    # Return true if it can be schedules, False otherwise
    dependencies = np.where(G[job] == 1)[0]
    if job in set_of_jobs:
        return False
    for dependency in dependencies:
        if dependency not in set_of_jobs:
            return False
    return True


def get_best_schedule_w_iterations():
    # Method without any modifications, nor iteration limitations
    iterations = 0
    q = PriorityQueue()

    # Keep the longest schedule
    longest_schedule: List = []
    longest_schedule_tardiness = 1000000
    longest_set_of_jobs = set()

    # Used for fathoming
    upper_bound = 1000000000

    # Add the possible initial jobs to the priority queue
    for j, processing_time, due_date in J:
        # Add jobs that are not prerequisites for other jobs,
        # i.e. jobs with no edges to other jobs, i.e. leaves
        dependencies = np.where(G[j] == 1)[0]
        if len(dependencies) == 0:
            tardiness = calc_tardiness_i([j])
            q.put((tardiness, [j], {j}))

    max_size_of_pending_list = q.qsize()

    while not q.empty() and iterations < 30000:
        # print(iterations)
        iterations += 1
        tardiness, list_of_jobs, set_of_jobs = q.get()
        if len(list_of_jobs) == no_of_jobs:
            return util_index2job(list(reversed(list_of_jobs)))

        # REMOVE COMMENTS TO GET CURRENT NODE AND TOTAL TARDINESS OF FIRST 2
        # AND LAST 2 ITERATIONS
        # if iterations <= 2 or iterations == 29999 or iterations == 29998:
        #     print(f'Current Node [{iterations}]:'
        #           f'{list(j + 1for j in reversed(list_of_jobs))}'
        #           f'\nTotal Tardiness:{tardiness}')

        for job in range(no_of_jobs):
            # Working in reversed order
            if job_can_be_scheduled(job, set_of_jobs):
                local_joblist = list_of_jobs.copy()
                local_jobset = set_of_jobs.copy()
                local_joblist.append(job)
                local_jobset.add(job)
                new_tardiness = calc_tardiness_i(local_joblist)

                # Fathoming
                if new_tardiness < upper_bound:
                    if len(local_joblist) == no_of_jobs:
                        if len(local_joblist) == no_of_jobs:
                            upper_bound = new_tardiness
                            fathoming(upper_bound, q)

                    q.put((new_tardiness, local_joblist, local_jobset))

                    # If there is a tie, this is a bit extra
                    if len(local_joblist) == len(longest_schedule):
                        # If the new tardiness is better than the old
                        if new_tardiness < longest_schedule_tardiness:
                            longest_schedule = local_joblist
                            longest_schedule_tardiness = new_tardiness
                            longest_set_of_jobs = local_jobset

                    elif len(local_joblist) > len(longest_schedule):
                        longest_schedule = local_joblist
                        longest_schedule_tardiness = new_tardiness
                        longest_set_of_jobs = local_jobset

                max_size_of_pending_list = max(max_size_of_pending_list,q.qsize())

    # Find missing jobs
    missing_jobs = set(range(no_of_jobs)) - longest_set_of_jobs
    while len(missing_jobs) > 0:
        ready = []
        for job in missing_jobs:
            if job_can_be_scheduled(job, longest_set_of_jobs):
                ready.append(job)

        # Append the job with the highest due date, higher chances of getting a non tardy job
        job_to_w_later_due_date = max(ready, key=lambda x: d[x])
        longest_schedule.append(job_to_w_later_due_date)
        missing_jobs.remove(job_to_w_later_due_date)
        longest_set_of_jobs.add(job_to_w_later_due_date)

    print(max_size_of_pending_list)
    return util_index2job(list(reversed(longest_schedule)))

def fathoming(upper_bound, pq: PriorityQueue):
    print('Begin Fathoming')
    to_append_back = []
    no_fathomed = 0
    while not pq.empty():
        tardiness, list_of_jobs, set_of_jobs = pq.get()
        if tardiness < upper_bound:
            to_append_back.append((tardiness, list_of_jobs, set_of_jobs))
        else:
            # print(f'{tardiness} < {upper_bound}')
            no_fathomed += 1
    for entry in to_append_back:
        pq.put(entry)

    print(f'Number of nodes pruned: {no_fathomed}')


# def get_best_schedule():
#     # Method without any modifications, nor iteration limitations
#     iterations = 0
#     q = PriorityQueue()
#     upper_bound = 10000000
#
#     # Add the possible initial jobs to the priority queue
#     for j, processing_time, due_date in J:
#         # Add jobs that are not prerequisites for other jobs,
#         # i.e. jobs with no edges to other jobs, i.e. leaves
#         dependencies = np.where(G[j] == 1)[0]
#         if len(dependencies) == 0:
#             tardiness = calc_tardiness_i([j])
#             q.put((tardiness, [j], {j}))
#
#     while not q.empty():
#         print(iterations)
#         iterations += 1
#         tardiness, list_of_jobs, set_of_jobs = q.get()
#         if len(list_of_jobs) == no_of_jobs:
#             return util_index2job(list(reversed(list_of_jobs)))
#
#         for job in range(no_of_jobs):
#             # Working in reversed order
#             if job_can_be_scheduled(job, set_of_jobs):
#                 local_joblist = list_of_jobs.copy()
#                 local_jobset = set_of_jobs.copy()
#                 local_joblist.append(job)
#                 new_tardiness = calc_tardiness_i(local_joblist)
#
#                 # Check for pruning
#                 if new_tardiness < upper_bound:
#                     local_jobset.add(job)
#                     # If this is a trial solution:
#                     if len(local_joblist) == no_of_jobs:
#                         upper_bound = new_tardiness
#                         fathoming(upper_bound, q)
#
#                     q.put((new_tardiness, local_joblist, local_jobset))


# def get_best_schedule_beam(beam):
#     # Get best schedule with beam
#     q = PriorityQueue()
#
#     # Add the possible initial jobs to the priority queue
#     for j, processing_time, due_date in J:
#         # Add jobs that are not prerequisites for other jobs,
#         # i.e. jobs with no edges to other jobs, i.e. leaves
#         dependencies = np.where(G[j] == 1)[0]
#         if len(dependencies) == 0:
#             tardiness = calc_tardiness_i([j])
#             q.put((tardiness, [j], {j}))
#
#     iterations = 0
#     while not q.empty() and iterations < 30000:
#         # print(iterations)
#         iterations += 1
#
#         # Best schedule so far, with least tardiness
#         # Important! List of jobs has the jobs in reverse order: i.e. from last to first
#         tardiness, list_of_jobs, set_of_jobs = q.get()
#
#         # If all jobs have been allocated (and it is the minimum), we have found the best schedule
#         if len(list_of_jobs) == no_of_jobs:
#             return [j_i for j_i in reversed(list_of_jobs)]
#
#         # Stop the loop if this many schedules have been added
#         counter = beam
#         # Sort the jobs from soonest to latest
#         due_dates_sorted_idxs = np.argsort(d)
#         idx = 0
#         # If job has already been scheduled, then skip
#         while counter > 0 and idx < len(due_dates_sorted_idxs) \
#                 and due_dates_sorted_idxs[idx] not in set_of_jobs:
#             job = due_dates_sorted_idxs[idx]
#             # Working in reversed order
#
#             # For a job to be scheduled, the jobs that depend on it need to already been scheduled, otherwise it will
#             # schedule jobs that lead to a sequence that does not respect precedence
#             # dependencies = np.where(G[job] == 1)[0]
#             if job_can_be_scheduled(job, set_of_jobs):
#                 local_joblist = list_of_jobs.copy()
#                 local_jobset = set_of_jobs.copy()
#                 local_joblist.append(job)
#                 new_tardiness = calc_tardiness_i(local_joblist)
#                 # new_tardiness = new_tardiness + 0.075 * new_tardiness / len(local_joblist) * (
#                 #         no_of_jobs - len(local_joblist))
#                 local_jobset.add(job)
#                 q.put((new_tardiness, local_joblist, local_jobset))
#                 counter -= 1
#
#             idx += 1

### Tests on utility methods
# # Check util for feasible jobs
# dummy_schedule = [30, 4, 3]
# assert util_is_feasible(dummy_schedule), 'Schedule is feasible'
# dummy_schedule = [4, 30, 3]
# assert not util_is_feasible(dummy_schedule), 'Schedule is feasible'
#
# # Check if the schedule is complete
# dummy_schedule = list(range(1,32))
# assert util_is_complete(dummy_schedule), 'Schedule is complete'
# dummy_schedule = list(range(1,32))
# dummy_schedule.remove(2)
# assert not util_is_complete(dummy_schedule), 'Schedule is not complete'

# Test can be scheduled
# assert job_can_be_scheduled(1, {30,0}), 'Job can be scheduled'
# assert not job_can_be_scheduled(1, {30}), 'Job cannot be scheduled'

# Test Hu's
# distance = calc_hus_heuristic()
# print(distance)

# assert distances[30] == 0, 'Incorrect distance for job 31'
# assert len(distances) == no_of_jobs, f'Incorrect number of jobs: {len(distances)}'

# from timeit import Timer
# timer = Timer("""get_best_schedule_w_iterations()""")
# print(timer.timeit())

# import time

# get the start time

# main program
# find sum to first 1 million numbers

# get the end time
# et = time.time()

# get the execution time
# elapsed_time = et - st
# print('Execution time:', elapsed_time, 'seconds')

# schedule = get_best_schedule_w_iterations()
# print(schedule)
# print(calc_tardiness(schedule, True))
# assert util_is_feasible(util_index2job(schedule)), 'Schedule should be feasible'
# assert util_is_complete(util_index2job(schedule)), 'Schedule should be complete'

# print(J)
# schedule = get_best_schedule()
# print(schedule)
# print(calc_tardiness(schedule, True))

# jan = [30, 4, 10, 3, 23, 14, 20, 22, 21, 19, 18, 9, 8, 7, 6, 17, 16, 29, 28, 27, 26, 25, 24, 15, 13, 12, 11, 5, 2, 1, 31]
# jan = [j_i - 1 for j_i in jan]

# our = [29, 3, 2, 22, 21, 20, 19, 18, 17, 13, 9, 8, 7, 6, 5, 16, 15, 28, 27, 26, 25, 24, 23, 14, 12, 11, 10, 4, 1, 0, 30]

# print(calc_tardiness_i(jan))

# print(calc_tardiness_i(our, True))
# print(calc_tardiness_i(jan, True))

# Hus Heuristic
schedule = get_best_schedule_w_heuristic()
print(schedule)
assert util_is_feasible(schedule), 'Schedule must be feasible'
print(calc_tardiness(schedule, True))