from datetime import datetime
import sys
import numpy as np
import os

action = sys.argv[1]


if action == "w":
    f = open("aux.txt", "w+")
    f.close()

elif action == "r":
    method = sys.argv[2]
    results = []
    f = open("aux.txt", "r")
    for y in f.read().split('\n'):
        if y.isdigit():
            results.append(float(y))
    os.remove("aux.txt")
    score = np.mean(results)
    now = datetime.now()
    current_time = now.strftime("%m, %d, %y, %H:%M:%S")
    r = open("results.txt", "a")
    r.write(f"Time: {current_time}, Method:{method}, Cost_of_solution: {score}\n")

    r.close()
