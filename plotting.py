import numpy
import matplotlib.pyplot as plt
import pickle

file = open('data/thompson_plot_data_bak.pkl', 'rb')
woRejection, wRejection = pickle.load(file)

steps = range(len(woRejection[0]))

woRejection, wRejection = numpy.array(woRejection), numpy.array(wRejection)
woRejection_mean, wRejection_mean = woRejection.mean(axis=0), wRejection.mean(axis=0)
woRejection_std, wRejection_std = woRejection.std(axis=0), wRejection.std(axis=0)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(steps, woRejection_mean, color='blue', label='without rejection')
ax.plot(steps, wRejection_mean, color='red', label='with rejection')

ax.fill_between(steps, woRejection_mean - woRejection_std, woRejection_mean + woRejection_std, color='blue', alpha=0.2)
ax.fill_between(steps, wRejection_mean - wRejection_std, wRejection_mean + wRejection_std, color='red', alpha=0.2)

ax.legend()
plt.show()
