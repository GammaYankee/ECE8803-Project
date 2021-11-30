import numpy


def ThompsonSampling(n_arms):
    # Does Thompson sampling with beta priors.
    try:
        alphas = [1 for i in range(n_arms)]
        betas = [1 for i in range(n_arms)]

        while True:
            # Generate a sample from each arm, and then guess the best one.
            sample = numpy.random.beta(alphas, betas)
            guess = max([(s, i) for i, s in enumerate(sample)])[1]
            yield guess

            # Get the next reward
            reward = yield
            if reward == 1:
                alphas[guess] += 1
            else:
                betas[guess] += 1
    except GeneratorExit:
        # Clean up after generator finishes
        pass


def ThompsonSamplingwRejection(arm_estimates, arm_tolerances):
    # Does Thompson sampling, but where arm_estimates[i] gives an estimate of
    # the mean of arm i, and arm_tolerances[i] gives a range that arm_estimates[i] can be off by.
    try:
        n_arms = len(arm_estimates)
        alphas = [1 for i in range(n_arms)]
        betas = [1 for i in range(n_arms)]

        while True:
            # Generate a sample from each arm, and then guess the best one.
            samples = []
            for i in range(n_arms):
                while True:
                    sample = numpy.random.beta(alphas[i], betas[i])
                    if abs(sample - arm_estimates[i]) <= arm_tolerances[i]:
                        samples.append(sample)
                        break

            guess = max([(s, i) for i, s in enumerate(samples)])[1]
            yield guess

            # Get the next reward
            reward = yield
            if reward == 1:
                alphas[guess] += 1
            else:
                betas[guess] += 1
    except GeneratorExit:
        # Clean up after generator finishes
        pass
