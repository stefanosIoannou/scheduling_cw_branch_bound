
from queue import PriorityQueue
import numpy as np
from task import *


# Processing Times
vii :        19.2204
blur :       5.5903
night :      22.7764
onnx :       2.9165
emboss :     1.7308
muse :       14.7329
wave :       10.6599

q = PriorityQueue()
# entry structure: (tardiness, [list of jobs], set(jobs already scheduled))

def nonRecursiveTopologicalSortUtil(v, visited, stack):
    # working stack contains key and the corresponding current generator
    working_stack = [v]

    while working_stack:
        # get last element from stack
        job = working_stack.pop()
        visited[v] = True

        # run through neighbor generator until it's empty
        dependencies = np.where(G[job] == 1)[0]
        for dependencie in dependencies:
            if not visited[dependencie]:  # not seen before?
                # remember current work
                working_stack.append(v)
                # restart with new neighbor
                working_stack.append(list(range(no_of_jobs))[dependencie])
                break
        else:
            # no already-visited neighbor (or no more of them)
            stack.append(v)


# The function to do Topological Sort.
def nonRecursiveTopologicalSort(jobs_to_order, jobs_already_ordered):
    # Remove the dependencies of jobs already ordered
    for job in jobs_already_ordered:
        G[job] = 0

    # Mark all the vertices as not visited
    visited = [False]*len(jobs_to_order)

    # result stack
    stack = []

    # Call the helper function to store Topological
    # Sort starting from all vertices one by one
    for i in range(jobs_to_order):
        if not (visited[i]):
            nonRecursiveTopologicalSortUtil(i, visited, stack)
    # Print contents of the stack in reverse
    stack.reverse()
    print(stack)


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


def tardy_so_far(list_of_job_indexes):
    tardy = 0
    end_time = total_processing_time
    for j in list_of_job_indexes:
        job = J[j]
        tardy += 1 if job[2] < end_time else 0
        end_time -= job[1]
    return tardy




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
        # tardiness = calc_tardiness_i([j])
        tardiness = tardy_so_far([j]) # TODO: remove this
        q.put((tardiness, [j], {j}))


def edd(jobs):
    missing_jobs = set(jobs)
    longest_set_of_jobs = set(range(no_of_jobs)) - missing_jobs
    longest_schedule = []
    def dependencies_fulfilled(job, set_of_jobs):
        dependencies = np.where(G[job] == 1)[0]
        for dependencie in dependencies:
            if dependencie not in set_of_jobs:
                return False
        return True
    while len(missing_jobs) > 0:
        ready = []
        for job in missing_jobs:
            if dependencies_fulfilled(job, longest_set_of_jobs):
                ready.append(job)
        for job in sorted(ready, key=lambda x: d[x], reverse=True):
            longest_schedule.append(job)
            missing_jobs.remove(job)
            longest_set_of_jobs.add(job)
            break
    longest_schedule.reverse()
    return longest_schedule
def moore_hodgson(jobs):
    # jobs = sorted(jobs, key=lambda x: d[x], reverse=False)
    on_time_schedule = []
    tardy_schedule = []
    processing_time = 0

    for job in edd(jobs):
        on_time_schedule.append(job)
        processing_time += p[job]
        if processing_time > d[job]:
            # From on_time_schedule, remove the job with the highest processing time
            # and add it to tardy_schedule
            longest_job = max(on_time_schedule, key=lambda x: p[x])
            on_time_schedule.remove(longest_job)
            tardy_schedule.append(longest_job)
            processing_time -= p[longest_job]

    # return on_time_schedule + tardy_schedule
    return len(tardy_schedule)

def get_best_schedule():
    def get_best_schedule_i(upper_bound=float('inf'), heuristic=False):
        i = 0
        longest_schedule = []
        longest_schedule_tardiness = float('inf')
        longest_set_of_jobs = set()

        q = PriorityQueue()
        for j, processing_time, due_date in J:
            # print('This is a job', j)
            # print(tardiness)
            # Bit Magic
            # bin_set = 1
            # bin_set &= ~(1 << j)

            # Add jobs that are not prerequisites for other jobs
            dependencies = np.where(G[j] == 1)[0]
            if len(dependencies) == 0:
                # tardiness = calc_tardiness_i([j])
                tardiness = tardy_so_far([j])  # TODO: remove this
                q.put((tardiness, [j], {j}))

        # while not q.empty() and i < 30000:
        while not q.empty():
            print(i)
            i += 1

            # Get the front of the queue
            tardiness, list_of_jobs, set_of_jobs = q.get()
            # If all jobs have been allocated (and it is the minimum), we have found the best schedule
            if len(list_of_jobs) == no_of_jobs:
                return [j_i for j_i in reversed(list_of_jobs)]

            # counter = 3
            # for job in reversed(np.argsort(d)):
            for job in range(no_of_jobs):
                # if counter <= 0:
                #     break
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
                # Get the longest_job, use total tardiness to break ties
                # Normal tardiness
                # new_tardiness = calc_tardiness_i(local_joblist)
                # Number of tardy jobs
                # new_tardiness = tardy_so_far(local_joblist) # TODO: remove this
                # missing_jobs = set(range(no_of_jobs)) - local_jobset
                # new_tardiness += moore_hodgson(missing_jobs)

                # Tardiness heuristic by proportion of tardy jobs
                # new_tardiness = calc_tardiness_i(local_joblist)
                # new_tardiness += tardy_so_far(local_joblist)/len(local_joblist)*(no_of_jobs-len(local_joblist))*(new_tardiness/len(local_joblist))

                # Tardiness heuristic by proportion of tardy jobs and moore hodgson - seems promising
                # new_tardiness = calc_tardiness_i(local_joblist)
                # missing_jobs = set(range(no_of_jobs)) - local_jobset
                # new_tardiness += moore_hodgson(missing_jobs)*(new_tardiness/len(local_joblist))

                # Heuristics
                new_tardiness = calc_tardiness_i(local_joblist)
                if heuristic:
                    new_tardiness = new_tardiness + 0.5 * new_tardiness / len(local_joblist) * (no_of_jobs - len(local_joblist))

                if new_tardiness > upper_bound:
                    continue

                if len(local_joblist) == len(longest_schedule):
                    # Check the tardiness
                    if new_tardiness < longest_schedule_tardiness:
                        longest_schedule = local_joblist
                        longest_schedule_tardiness = new_tardiness
                        longest_set_of_jobs = local_jobset

                elif len(local_joblist) > len(longest_schedule):
                    longest_schedule = local_joblist
                    longest_schedule_tardiness = new_tardiness
                    longest_set_of_jobs = local_jobset

                local_jobset.add(job)
                q.put((new_tardiness, local_joblist, local_jobset))
                # counter -= 1

        # Last resort: We have the greedy less tardy solution, and longest
        missing_jobs = set(range(no_of_jobs)) - longest_set_of_jobs

        def dependencies_fulfilled(job, set_of_jobs):
            dependencies = np.where(G[job] == 1)[0]
            for dependencie in dependencies:
                if dependencie not in set_of_jobs:
                    return False
            return True
        print(longest_schedule)
        print(missing_jobs)
        while len(missing_jobs) > 0:
            ready = []
            for job in missing_jobs:
                if dependencies_fulfilled(job, longest_set_of_jobs):
                    ready.append(job)
            for job in sorted(ready, key=lambda x: d[x], reverse=True):
                longest_schedule.append(job)
                missing_jobs.remove(job)
                longest_set_of_jobs.add(job)
                break

        return [j_i for j_i in reversed(longest_schedule)]
    schedule = get_best_schedule_i(heuristic=True)
    upper_bound = calc_tardiness_i(schedule, True)
    return get_best_schedule_i(upper_bound)






# print(J)
print(calc_tardiness_i(edd(list(range(no_of_jobs))), True))
schedule = get_best_schedule()
print(schedule)
print(calc_tardiness_i(schedule, True))
jan = [30, 4, 10, 3, 23, 14, 20, 22, 21, 19, 18, 9, 8, 7, 6, 17, 16, 29, 28, 27, 26, 25, 24, 15, 13, 12, 11, 5, 2, 1, 31]
jan = [j_i - 1 for j_i in jan]

our = [29, 3, 2, 22, 21, 20, 19, 18, 17, 13, 9, 8, 7, 6, 5, 16, 15, 28, 27, 26, 25, 24, 23, 14, 12, 11, 10, 4, 1, 0, 30]

# print(calc_tardiness_i(jan))

print(calc_tardiness_i(our, True))
print(calc_tardiness_i(jan, True))
