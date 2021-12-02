import numpy
from simulation import simulate_mean_width_instances
import multiprocessing
from multiprocessing import Pool
import pickle


def main_simulation():
    mu1_est = 0.6
    n_rounds = 200
    n_steps = 15
    # generate mesh
    width = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    delta = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    test_cases = []

    for i in range(len(width)):
        for j in range(len(delta)):
            width_i, delta_j = width[i], delta[j]
            if 2 * width_i > delta_j:
                test_cases.append([width_i, delta_j, mu1_est, n_rounds, n_steps])

    print("Number of CPU:", multiprocessing.cpu_count())

    p = Pool(4)
    p.map(simulate_mean_width_instances, test_cases)

    file = open('data/experiments_{}/test_cases.pkl'.format(mu1_est), 'wb')
    pickle.dump(test_cases, file)


if __name__ == "__main__":
    main_simulation()
