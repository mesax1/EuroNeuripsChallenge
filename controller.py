# Controller has an environment and tests it against a dynamic solver program
import subprocess
import argparse
import tools
import json
import sys
import numpy as np
import threading
from environment import VRPEnvironment
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", help="Instance to solve")
    parser.add_argument("--instance_seed", type=int, default=1, help="Seed to use for the dynamic instance")
    parser.add_argument("--static", action='store_true', help="Add this flag to solve the static variant of the problem (by default dynamic)")
    parser.add_argument("--epoch_tlim", type=int, default=120, help="Time limit per epoch")
    parser.add_argument("--timeout", type=int, default=3600, help="Global timeout (seconds) to use")

    try:
        split_idx = sys.argv.index("--")
    except ValueError:
        print("Usage: python controller.py {options} -- {solver}")
        sys.exit()

    args = parser.parse_args(sys.argv[1:split_idx])
    solver_cmd = sys.argv[split_idx+1:]

    # Load instance
    entries = os.listdir('instances/')
    static_instance = tools.read_vrplib(f"instances/{entries[int(args.instance)]}")
    #static_instance = tools.read_vrplib(args.instance)

    # Create environment
    env = VRPEnvironment(args.instance_seed, static_instance, args.epoch_tlim, args.static)

    done = False

    # Start subprocess and interact with it
    with subprocess.Popen(solver_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as p:

        # Set global timeout, bit ugly but this will also interrupt waiting for input
        timeout_timer = threading.Timer(args.timeout, lambda: p.kill())
        timeout_timer.daemon = True
        timeout_timer.start()

        for line in p.stdout:
            line = line.strip()
            request = json.loads(line)
            if request['action'] == 'step':
                solution = [np.array(route) for route in request['data']]
                observation, reward, done, info = env.step(solution)

                response = dict(
                    observation=observation,
                    reward=reward,
                    done=done,
                    info=info
                )
            elif request['action'] == 'reset':
                assert env.reset_counter == 0, "Can only reset environment once!"
                observation, info = env.reset()
                response = dict(
                    observation=observation,
                    info=info
                )
            else:
                raise Exception("Invalid request")
            
            response_str = tools.json_dumps_np(response)
            p.stdin.write(response_str)
            p.stdin.write('\n')
            p.stdin.flush()

        # Cancel timer (does nothing if timer already triggered) and wait for it to finish
        timeout_timer.cancel()
        timeout_timer.join()

        # Catch remaining output and wait at most 10 secs for solver thread to finish gracefully
        return_code = p.wait(10)
        assert return_code == 0, "Solver did not exit succesfully"

    assert done, "Environment is not finished"
    # Write results
    print(f"------ Controller ------")
    print(f"Cost of solution: {sum(env.final_costs.values())}")
    print("Solution:")
    print(tools.json_dumps_np(env.final_solutions))

    #f = open("aux.txt", "a")
    #f.write(f"{sum(env.final_costs.values())}\n")
    #f.close()

    csv = open("instance_info.txt", "a")
    if solver_cmd[-1][:2] == "f2":
        strategy = "f2"
        alpha = solver_cmd[-1][2:]
    elif solver_cmd[-1][:19] == "knearestimedistance":
        strategy = "knearestimedistance"
        alpha = solver_cmd[-1][19:]
    elif solver_cmd[-1][:16] == "modifiedknearest":
        string = solver_cmd[-1].split(",")
        strategy = string[0]
        c = string[1]
        alpha = string[2]
        beta = string[3]
        k = string[4]
        omega = string[5]
    elif solver_cmd[-1][:20] == "removeorderedclients":
        string = solver_cmd[-1].split(",")
        strategy = string[0]
        gamma = string[1]

    else:
        strategy = solver_cmd[-1]
        # c = "NaN"
        # alpha = "NaN"
        # beta = "NaN"
        # k = "NaN"
        # omega = "NaN"
        gamma = "NaN"
    # csv.write(f"\n{entries[int(args.instance)]};{args.instance};{sum(env.final_costs.values())};{strategy};c={c};"
    #           f"aplha={alpha};beta={beta};k={k};omega={omega}")
    csv.write(f"\n{entries[int(args.instance)]};{args.instance};{sum(env.final_costs.values())};{strategy};gamma={gamma}")
    csv.close()
