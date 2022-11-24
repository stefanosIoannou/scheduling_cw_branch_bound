from queue import PriorityQueue
import numpy as np
from task import *
from collections import deque
import argparse


def calculate_hus_heuristic():
    """
    Generates Hu's heuristic. Each job is given its Hu's value.

    :return: dictionary from job to heuristic
    """

    def calculate_length_recursively(job):
        """
        Calculate recursively the Hu's heuristic for a job.

        :param job: job to calculate
        :return: heuristic's value
        """
        if np.sum(G[job]) == 0:
            return 1
        else:
            return 1 + max([calculate_length_recursively(child) for child in np.where(G[job] == 1)[0]])

    path_length = {}
    for j in range(no_of_jobs):
        path_length[j] = calculate_length_recursively(j)
    return path_length


def get_best_schedule_w_heuristic(verbose=False):
    """
    Schedules jobs according to Hu's algorithm

    :return: Complete schedule
    """

    hue_distances = calculate_hus_heuristic()
    schedule = []
    missing_jobs = set(range(no_of_jobs))
    iterations = 0
    while len(missing_jobs) > 0:
        iterations += 1
        ready = []
        for job in missing_jobs:
            if job_can_be_scheduled(job, set(schedule)):
                ready.append(job)
        # Hu algorithm schedules first jobs with the highest hue distance, but we are working in reverse. So we get the
        # job with the lowest hue distance and brake ties with the highest job index
        min_job = min(ready, key=lambda x: (hue_distances[x], -x))

        if verbose:
            with open('bnb_hus_partial_solutions.txt', 'a' if iterations > 1 else 'w') as f:
                f.write(f'---------------------------------------------------------------Iter:[{iterations}]:\n'
                        f'    Current Node: {util_index2job(reversed(schedule))}\n'
                        f'    Node Tardiness: {calc_tardiness_i(missing_jobs)}\n')

        schedule.append(min_job)
        missing_jobs.remove(min_job)

    return util_index2job(reversed(list(schedule)))


def calc_tardiness_i(list_of_job_indexes, reverse=False):
    """
    Calculate the total tardiness of a list of job indexes.

    :param list_of_job_indexes: List of job indexes to calculate the total tardiness of.
    :param reverse: True if the list of jobs is not reversed, i.e. the schedule goes from first job to last job.
                    False otherwise.
    :return: The total tardiness
    """
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


def calc_tardiness(schedule):
    """
    Calculate the tardiness of a schedule

    :param schedule: List of jobs
    :return: The total tardiness
    """
    # Convert to indexes
    indexes = [j - 1 for j in schedule]
    return calc_tardiness_i(indexes, True)


def util_is_feasible(schedule):
    """
    Return true if the schedule is complete, i.e. all jobs have been added

    :param schedule: Schedule of jobs
    :return: True if the schedule is complete, False otherwise
    """
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
    """
    Return true if the schedule is complete, i.e. all jobs have been added

    :param schedule: Schedule of jobs
    :return: True if the schedule is complete, False otherwise
    """
    # Return if the schedule is complete
    return set(range(1, 32)) == set(schedule)


def util_index2job(schedule):
    """
    Return a list of jobs, from a list of job indexes

    :param schedule: Schedule of job indexes
    :return: Schedule of jobs
    """
    return [j + 1 for j in schedule]


def job_can_be_scheduled(job, set_of_jobs: set):
    """
    Return true if a job can be scheduled, given a set of jobs that were already scheduled.
    A job can be scheduled if all its children, i.e. the jobs it has an edge to, have already been added to the
    list (since the jobs are added to the front of the schedule). A job cannot be scheduled if it has already been
    scheduled. Return false otherwise.

    :param job: job index to be added
    :param set_of_jobs: a set of jobs already scheduled
    :return: True if the job can be scheduled, False otherwise
    """
    # Check it the job can be schedules, i.e. the children of the job are already in the job_schedule.
    # Return true if it can be schedules, False otherwise
    dependencies = np.where(G[job] == 1)[0]
    if job in set_of_jobs:
        return False
    for dependency in dependencies:
        if dependency not in set_of_jobs:
            return False
    return True


def get_best_schedule_w_iterations(verbose=False):
    """
    Run Branch and Bound algorithm for up to 30K iterations (Question 2). The solution from the algorithm is
    made complete and feasible. During the first two and last two iterations, the current node and
    the total tardiness of those nodes is reported. Add the end of the computation the maximum size of the
    pending list is reported. A schedule is returned.

    :param verbose: Set to True, to print the partial solution, and the total tardiness at each iteration
    :return: Schedule found
    """
    # Method without any modifications, nor iteration limitations
    iterations = 0
    q = PriorityQueue()

    # Keep the longest schedule
    longest_schedule = []
    longest_schedule_tardiness = float('inf')
    longest_set_of_jobs = set()

    # Used for fathoming
    upper_bound = float('inf')

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
        iterations += 1
        tardiness, list_of_jobs, set_of_jobs = q.get()
        if len(list_of_jobs) == no_of_jobs:
            return util_index2job(list(reversed(list_of_jobs)))

        if verbose:
            with open('bnb_partial_solutions.txt', 'a' if iterations > 1 else 'w') as f:
                f.write(f'---------------------------------------------------------------Iter:[{iterations}]:\n'
                        f'    Current Node: {util_index2job(reversed(list_of_jobs))}\n'
                        f'    Node Tardiness: {tardiness}\n')

        # REMOVE COMMENTS TO GET CURRENT NODE AND TOTAL TARDINESS OF FIRST 2
        # AND LAST 2 ITERATIONS
        if iterations <= 2 or iterations == 29999 or iterations == 29998:
            print(f'Iter:[{iterations}]:\n'
                  f'    Current Node: {util_index2job(reversed(list_of_jobs))}\n'
                  f'    Node Tardiness: {tardiness}\n')

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

                max_size_of_pending_list = max(max_size_of_pending_list, q.qsize())

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

    print(f'max size of the pending list : {max_size_of_pending_list}')
    return util_index2job(list(reversed(longest_schedule)))


def fathoming(upper_bound, pq):
    """
    Perform fathoming on the pending list (priority queue), based on the upper bound

    :param upper_bound: The upper bound. Any tardiness higher than this value is removed from the priority queue
    :param pq: Priority Queue. The fathoming is done in place
    :return: Fathomed Priority Queue
    """
    print('Begin Fathoming')
    to_append_back = []
    no_fathomed = 0
    while not pq.empty():
        tardiness, list_of_jobs, set_of_jobs = pq.get()
        if tardiness < upper_bound:
            to_append_back.append((tardiness, list_of_jobs, set_of_jobs))
        else:
            no_fathomed += 1
    for entry in to_append_back:
        pq.put(entry)

    print(f'Number of nodes pruned: {no_fathomed}')


def fathoming_stack(upper_bound, stack):
    """
    Perform fathoming on the pending list (stack), based on the upper bound

    :param upper_bound: The upper bound. Any tardiness higher than this value is removed from the stack
    :param stack: Stack. A new stack is created
    :return: Fathomed Priority stack
    """
    print('Begin Fathoming')
    size = len(stack)
    stack_filtered = list(filter(lambda x: x[0] < upper_bound, stack))

    print(f'Number of nodes pruned: {size - len(stack_filtered)}')
    return stack_filtered


def tardy_so_far(list_of_job_indexes):
    """
    Measures how many tardy jobs there are in a schedule

    :param list_of_job_indexes: jobs already scheduled
    :return: number of tardy jobs
    """
    tardy = 0
    end_time = sum(p)
    for j in list_of_job_indexes:
        job = J[j]
        tardy += 1 if job[2] < end_time else 0
        end_time -= job[1]
    return tardy


def edd(jobs):
    """
    Generated a valid Earliest Due Date list of jobs

    :param jobs: list of jobs to perform EDD
    :return: valid schedule
    """
    missing_jobs = set(jobs)
    longest_set_of_jobs = set(range(no_of_jobs)) - missing_jobs
    longest_schedule = []

    while len(missing_jobs) > 0:
        ready = []
        for job in missing_jobs:
            if job_can_be_scheduled(job, longest_set_of_jobs):
                ready.append(job)

        job = max(ready, key=lambda x: d[x])
        longest_schedule.append(job)
        missing_jobs.remove(job)
        longest_set_of_jobs.add(job)

    longest_schedule.reverse()
    return longest_schedule


def moore_hodgson(jobs):
    """
    Applies Moore-Hodgson to measure how many tardy jobs there are if jobs are scheduled

    :param jobs: list of jobs to schedule
    :return: number of tardy jobs
    """
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

    return len(tardy_schedule)


def get_best_schedule(verbose=False, heuristic='none', alpha=0.075):
    """
    Run Branch and Bound algorithm without any limitation to the number of iterations.
    The schedule with the global optimum tardiness is returned.

    :param verbose: Set to True, to print the partial solution, and the total tardiness at each iteration
    :param heuristic: heuristic to apply ("proportion" or "moore_hodgson"), none by default
    :param alpha: if heuristic "proportion" chosen, alpha value to apply between [0,1], 0.075 by default
    :return: global optimum schedule
    """
    iterations = 0
    q = PriorityQueue()
    upper_bound = float('inf')
    no_of_jobs = len(J)

    # Add the possible initial jobs to the priority queue
    for j, processing_time, due_date in J:
        # Add jobs that are not prerequisites for other jobs,
        # i.e. jobs with no edges to other jobs, i.e. leaves
        dependencies = np.where(G[j] == 1)[0]
        if len(dependencies) == 0:
            tardiness = calc_tardiness_i([j])
            q.put((tardiness, [j], {j}))

    while not q.empty():
        iterations += 1
        tardiness, list_of_jobs, set_of_jobs = q.get()
        if len(list_of_jobs) == no_of_jobs:
            print(f'Iterations: {iterations}]')
            return util_index2job(list(reversed(list_of_jobs)))

        if verbose:
            with open('bnb_unbounded.txt', 'a' if iterations > 1 else 'w') as f:
                f.write(f'---------------------------------------------------------------Iter:[{iterations}]:\n'
                        f'    Current Node: {util_index2job(reversed(list_of_jobs))}\n'
                        f'    Node Tardiness: {tardiness}\n')

        for job in range(no_of_jobs):
            # Working in reversed order
            if job_can_be_scheduled(job, set_of_jobs):
                local_joblist = list_of_jobs.copy()
                local_jobset = set_of_jobs.copy()
                local_joblist.append(job)
                local_jobset.add(job)
                new_tardiness = calc_tardiness_i(local_joblist)
                # If heuristic method is chosen, it modifies the tardiness to according to the heuristic
                if heuristic == 'moore_hodgson':
                    missing_jobs = set(range(no_of_jobs)) - local_jobset
                    new_tardiness += moore_hodgson(missing_jobs) * (new_tardiness / len(local_joblist))
                elif heuristic == 'proportion':
                    assert 0 <= alpha <= 1, 'alpha must be between 0 and 1'
                    new_tardiness += alpha * new_tardiness / len(local_joblist) * (no_of_jobs - len(local_joblist))

                # Check for pruning
                if new_tardiness < upper_bound:
                    # If this is a trial solution:
                    if len(local_joblist) == no_of_jobs:
                        upper_bound = new_tardiness
                        fathoming(upper_bound, q)

                    q.put((new_tardiness, local_joblist, local_jobset))


def get_best_schedule_dfs(verbose=False):
    """
    Apply DFS to find the optimal schedule. Uses upperbound and fathoming to speed prune the stack.

    :return: global optimum schedule
    """
    iterations = 0
    s = []
    subschedules_tardiness = {}
    upper_bound = float('inf')

    # Add the possible initial jobs to the stack
    for j, processing_time, due_date in J:
        # Add jobs that are not prerequisites for other jobs,
        # i.e. jobs with no edges to other jobs, i.e. leaves
        dependencies = np.where(G[j] == 1)[0]
        if len(dependencies) == 0:
            tardiness = calc_tardiness_i([j])
            s.append((tardiness, [j], {j}))

    best_schedule = []
    while len(s) > 0:
        iterations += 1
        tardiness, list_of_jobs, set_of_jobs = s.pop()
        if len(list_of_jobs) == no_of_jobs:
            if tardiness <= upper_bound:
                upper_bound = tardiness
                best_schedule = list_of_jobs

        if verbose:
            with open('bnb_dfs_partial_solution.txt', 'a' if iterations > 1 else 'w') as f:
                f.write(f'---------------------------------------------------------------Iter:[{iterations}]:\n'
                        f'    Current Node: {util_index2job(reversed(list_of_jobs))}\n'
                        f'    Node Tardiness: {tardiness}\n')

        # Instead of adding all the jobs one by one, we batch them and add them after expanding
        jobs_to_add = []
        for job in range(no_of_jobs):
            # Working in reversed order
            if job_can_be_scheduled(job, set_of_jobs):
                local_joblist = list_of_jobs.copy()
                local_jobset = set_of_jobs.copy()
                local_joblist.append(job)
                new_tardiness = calc_tardiness_i(local_joblist)

                # Check for pruning
                if new_tardiness < upper_bound:
                    local_jobset.add(job)
                    # If this is a trial solution:
                    if len(local_joblist) == no_of_jobs:
                        upper_bound = new_tardiness
                        s = fathoming_stack(upper_bound, s)

                    if subschedules_tardiness.get(frozenset(local_joblist), float('inf')) > new_tardiness:
                        jobs_to_add.append((new_tardiness, local_joblist, local_jobset))
                        subschedules_tardiness[frozenset(local_joblist)] = new_tardiness

        # Add the batched jobs in order of the tardiness
        jobs_to_add.sort(key=lambda x: x[0], reverse=True)
        s.extend(jobs_to_add)
    print("Iterations: ", iterations)
    return util_index2job(list(reversed(best_schedule)))


### Argparser
parser = argparse.ArgumentParser(
    prog='Scheduling Coursework',
    description='Run scheduling algorithms using processing times from q1 or q3')

parser.add_argument('-q', '--question', choices=[1, 3], type=int, required=True,
                    help='1 for question 1 processing times, '
                         '3 for question 3 processing times')
parser.add_argument('--algo', type=str, choices=['bnb_unbounded', 'bnb', 'bnb_hus', 'bnb_dfs'],
                    required=True, help='Algorithm to run')
parser.add_argument('-v', '--verbose', action='store_true', help='Print partial solutions for each iteration')

args = parser.parse_args()

if args.question == 1:
    print('Using processing times from question 1')
    J, p, d = get_tuple_list_q1()
elif args.question == 3:
    print('Using processing times from question 3')
    J, p, d = get_tuple_list_q3()

verbose = args.verbose
schedule = None
if args.algo == 'bnb_unbounded':
    print('Running Branch and Bound Unbounded')
    schedule = get_best_schedule(verbose)
elif args.algo == 'bnb':
    print('Running Branch and Bound (Solution to Q2)')
    schedule = get_best_schedule_w_iterations(verbose)
elif args.algo == 'bnb_hus':
    print('Running Branch and Bound with Hu\'s Heuristic (Solution to Q2)')
    schedule = get_best_schedule_w_heuristic(verbose)
elif args.algo == 'bnb_dfs':
    print('Running Branch and Bound with DFS (Solution to Q3)')
    schedule = get_best_schedule_dfs(verbose)
else:
    print('Nothing to run here')

print(f'Final Schedule: {schedule}\n'
      f'Total Tardiness: {calc_tardiness(schedule)}')
