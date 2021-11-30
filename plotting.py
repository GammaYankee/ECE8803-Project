import matplotlib.pyplot as plt
import pickle

file = open('data/thompson_plot_data.pkl', 'rb')
plot1, plot2 = pickle.load(file)

steps = range(len(plot1))

plt.plot(steps, plot1, label='without rejection')
plt.plot(steps, plot2, label='with rejection')

plt.legend()
plt.show()
