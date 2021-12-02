import numpy
from simulation import simulate
from thompson_sampling import ThompsonSampling, ThompsonSamplingwRejection
from ucb import UCBGenerator,  UCBGeneratorwConfidence
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


n_rounds = 40
n_steps = 20

woRejection = []
wRejection = []
ucb = []
ucbwConfidence = []
rounds = 15
mu1_est = 0.5
mu1_eps = 0.11
mu2_est = 0.7
mu2_eps = 0.11
for i in tqdm(range(20)):
    mu1 = numpy.random.uniform(mu1_est - mu1_eps, mu1_est + mu1_eps)
    mu2 = numpy.random.uniform(mu2_est - mu2_eps, mu2_est + mu2_eps)
    def addThing(x):
        return x[0].append(numpy.array(list(simulate(x[1], mu1, mu2,  2**rounds))))
    addThing((woRejection, ThompsonSampling(2)))
    addThing((wRejection, ThompsonSamplingwRejection([mu1_est, mu2_est], [mu1_eps, mu2_eps])))
    addThing((ucb, UCBGenerator(2, 2**rounds)))
    addThing((ucbwConfidence, UCBGeneratorwConfidence([mu1_est, mu2_est], [mu1_eps, mu2_eps], 2**rounds)))

file = open('data/thompson_plot_data.pkl', 'wb')
pickle.dump([woRejection, wRejection, ucb, ucbwConfidence], file)



