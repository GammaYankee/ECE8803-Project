import numpy
import math

def UCBGenerator(k, T):
    """Generator function for the UCB algorithm.
    This function can indefinitely take rewards
    as input and produces guesses as output.
    """
    try:
        # Set up variables
        totals = [0 for i in range(k)] # Total rewards for each arm
        counts = [0 for i in range(k)] # Counts of increments
        # For the first k rounds, guess the tth arm:
        for guess in range(k):
            yield guess
            counts[guess] += 1
            totals[guess] += yield
        # For remaining rounds, output a guess based on the UCB method.
        while True:
            # Generate a new guess
            # Note: I think the constant, as stated in the homework should be 4,
            # but I set it to 2,since this gives better results, more consistent with the textbook
            guess = max([(totals[i]/counts[i] + math.sqrt(math.log(T)/counts[i]), i) for i in range(k)])[1]
            yield guess
            # Get the next reward
            counts[guess] += 1
            totals[guess] += yield
    except GeneratorExit:
        # Clean up after generator finishes
        pass

def UCBGeneratorwConfidence(mu_est, mu_error, T):
    """Generator function for the UCB algorithm.
    This function can indefinitely take rewards
    as input and produces guesses as output.
    """
    try:
        k = len(mu_est)
        # Set up variables
        totals = [0 for i in range(k)] # Total rewards for each arm
        counts = [0 for i in range(k)] # Counts of increments
        # For the first k rounds, guess the tth arm:
        for guess in range(k):
            yield guess
            counts[guess] += 1
            totals[guess] += yield
        # For remaining rounds, output a guess based on the UCB method.
        while True:
            # Generate a new guess
            # Note: I think the constant, as stated in the homework should be 4,
            # but I set it to 2,since this gives better results, more consistent with the textbook
            guess = max([(min(totals[i]/counts[i] + math.sqrt(math.log(T)/counts[i]), mu_est[i]+mu_error[i]), i) for i in range(k)])[1]
            yield guess
            # Get the next reward
            counts[guess] += 1
            totals[guess] += yield
    except GeneratorExit:
        # Clean up after generator finishes
        pass
