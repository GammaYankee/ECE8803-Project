import numpy
from thompson_sampling import ThompsonSampling, ThompsonSamplingwRejection
from ucb import UCBGenerator, UCBGeneratorwConfidence
import pickle

def reward(guess, mu1, mu2):
    return numpy.random.binomial(1, (mu1, mu2)[guess])


def simulate(x, mu1, mu2, T):
    guesses = []
    guess = next(x)
    guesses.append(guess)
    iteration = 0
    next_response = 2
    while True:
        rew = reward(guess, mu1, mu2)
        next(x)  # Primes the generator to receive next input.
        guess = x.send(rew)
        guesses.append(guess)
        iteration += 1
        if iteration == next_response:
            next_response *= 2
            yield sum(max(mu1, mu2) - (mu1, mu2)[arm] for arm in guesses)
        if iteration == T:
            break


def simulate_mean_width_instances(test_case):
    mu1_eps = test_case[0]
    mu2_eps = test_case[0]
    delta = test_case[1]
    mu1_est = test_case[2]
    mu2_est = mu1_est - delta
    n_rounds = test_case[3]
    rounds = test_case[4]

    print('Simulating delta={}. width={}'.format(delta, mu1_eps))
    woRejection = []
    wRejection = []
    ucb = []
    ucbwConfidence = []

    for i in range(n_rounds):
        mu1 = numpy.random.uniform(mu1_est - mu1_eps, mu1_est + mu1_eps)
        mu2 = numpy.random.uniform(mu2_est - mu2_eps, mu2_est + mu2_eps)

        def addThing(x):
            return x[0].append(numpy.array(list(simulate(x[1], mu1, mu2, 2 ** rounds))))

        addThing((woRejection, ThompsonSampling(2)))
        addThing((wRejection, ThompsonSamplingwRejection([mu1_est, mu2_est], [mu1_eps, mu2_eps])))
        addThing((ucb, UCBGenerator(2, 2 ** rounds)))
        addThing((ucbwConfidence, UCBGeneratorwConfidence([mu1_est, mu2_est], [mu1_eps, mu2_eps], 2 ** rounds)))

    file = open('data/experiments_{}/thompson_data_delta{}_width{}.pkl'.format(mu1_est, delta, mu1_eps), 'wb')
    pickle.dump([woRejection, wRejection, ucb, ucbwConfidence], file)
