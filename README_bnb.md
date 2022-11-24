This is the Branch and Bound Coursework

```bash
# Run the Branch and Bound algorithm using: 
# the processing times of Q1 (Question 2 solution)
python3 main.py --q 1 --algo 'bnb'
# the processing times of Q3 (Q3 solution)
python3 main.py --q 3 --algo 'bnb'

# Run the Branch and Bound algorithm with 
# Hue's Heuristic with the processing times of question 2
# (Question 2 Solution)
python3 main.py --q 1 --algo 'bnb_hus'

# Run the unbounded version of Branch and Bound
# Using Q1 processing times
python3 main.py --q 1 --algo 'bnb_unbound'
# Using Q3 processing times
python3 main.py --q 3 --algo 'bnb_unbound'
```

To get a list of partial solutions use the flag `--verbose` or `-v`, with any command to generate a *.txt file, 
with the current node at each iteration and total tardiness.
