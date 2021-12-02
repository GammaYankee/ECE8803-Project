import numpy
import matplotlib.pyplot as plt
import pickle

file = open('data/thompson_plot_data.pkl', 'rb')
runs = [numpy.array(x) for x in pickle.load(file)]

steps = range(len(runs[0][0]))

names = ["TS without rejection", "TS with rejection", "UCB", "UCB with Confidence"]
colors = ["red", "green", "blue", "orange"]
means = [x.mean(axis=0) for x in runs]
stds = [x.std(axis=0) for x in runs]

fig, ax = plt.subplots(figsize=(8, 4))
for mean, std, name, color in zip(means, stds, names, colors):
    ax.plot(steps, mean, color=color, label=name)
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)

ax.legend()
plt.show()
