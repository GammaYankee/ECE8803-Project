import numpy
import matplotlib.pyplot as plt
import pickle
from utills import max_zero

mu_1_mean = 0.6
delta = 0.1
width = 0.1

file = open('plotting_data/experiments_{}/thompson_data_delta{}_width{}.pkl'.format(mu_1_mean, delta, width), 'rb')
runs = [numpy.array(x) for x in pickle.load(file)]

steps = [2 ** i for i in range(len(runs[0][0]))]

names = ["TS without rejection", "TS with rejection", "UCB", "UCB with Confidence"]
colors = ["red", "green", "blue", "orange"]
means = [x.mean(axis=0) for x in runs]
stds = [x.std(axis=0) for x in runs]


fig, ax = plt.subplots(figsize=(8, 5))
plt.xlabel('Time step', fontsize=15)
plt.ylabel('Regrets', fontsize=15)
# plt.title('Delta={}, width={}'.format(delta, width))
for mean, std, name, color in zip(means, stds, names, colors):
    ax.plot(steps, mean, color=color, label=name)
    ax.fill_between(steps, max_zero(mean - std), mean + std, color=color, alpha=0.1)

ax.legend()

fig.savefig('figures/regrets_delta{}_width{}.png'.format(delta, width))