import numpy


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
