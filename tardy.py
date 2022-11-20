from task import *
jobs = list(range(0, 31))
# p = [4, 17, 2, 2, 6, 2, 21, 6, 13, 6, 6, 2, 4, 4, 6, 13, 13, 13, 2, 4, 2, 4, 21, 6, 25, 17, 2, 4, 13, 2, 17]
# # processing time for question 2
# # p = [2.9165, 14.7329, 1.7308, 1.7308, 5.5903, 1.7308, 19.2204, 5.5903, 10.6599, 5.5903, 5.5903, 1.7308, 2.9165, 2.9165, 5.5903, 10.6599, 10.6599, 10.6599, 1.7308, 2.9165, 1.7308, 2.9165, 19.2204, 5.5903, 22.7764, 14.7329, 1.7308, 2.9165, 10.6599, 1.7308, 14.7329]
# d = [172, 82, 18, 61, 93, 71, 217, 295, 290, 287, 253, 307, 279, 73, 355, 34, 233, 77, 88, 122, 71, 181, 340, 141, 209, 217, 256, 144, 307, 329, 269]


def edd(jobs):
    missing_jobs = set(range(no_of_jobs))
    longest_set_of_jobs = set()
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

def tardy(jobs):
    jobs = sorted(jobs, key=lambda x: d[x], reverse=False)
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

print(tardy(jobs))
print(sum(p))
print(max(d))