import numpy
from simulation import simulate
from thompson_sampling import ThompsonSampling, ThompsonSamplingwRejection
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


n_rounds = 10
n_steps = 20


woRejection = []
wRejection = []
for i in tqdm(range(n_rounds)):
    mu1 = numpy.random.uniform(0.4, 0.6)
    mu2 = numpy.random.uniform(0.4, 0.6)

    woRejection.append(numpy.array(list(simulate(ThompsonSampling(2), mu1, mu2, 2 ** n_steps))))
    wRejection.append(
        numpy.array(list(simulate(ThompsonSamplingwRejection([0.5, 0.5], [0.1, 0.1]), mu1, mu2, 2 ** n_steps))))
plot1 = sum(woRejection) / n_rounds
plot2 = sum(wRejection) / n_rounds

file = open('data/thompson_plot_data.pkl', 'wb')
pickle.dump([plot1, plot2], file)



