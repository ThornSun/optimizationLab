import numpy
import random
import time
from Solution import Solution


def bat(maxiteration, population, dimension, objf, lb, ub):
    a = 0.5  # Loudness  (constant or decreasing)
    r = 0.5  # Pulse rate (constant or decreasing)
    qmin = 0  # Frequency minimum
    qmax = 2  # Frequency maximum
    # Initializing arrays
    q = numpy.zeros(population)  # Frequency
    v = numpy.zeros((population, dimension))  # Velocities
    Convergence_curve = []

    # Initialize the population/solutions
    Sol = numpy.random.rand(population, dimension) * (ub - lb) + lb
    S = numpy.copy(Sol)
    Fitness = numpy.zeros(population)

    # initialize solution for the final results   
    s = Solution()
    print("BAT is optimizing  \"" + objf.__name__ + "\"")

    # Initialize timer for the experiment
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Evaluate initial random solutions
    for i in range(0, population):
        Fitness[i] = objf(Sol[i, :])

    # Find the initial best solution
    fmin = min(Fitness)
    I = numpy.argmin(fmin)
    best = Sol[I, :]

    # Main loop
    for t in range(0, maxiteration):

        # Loop over all bats(solutions)
        for i in range(0, population):
            q[i] = qmin + (qmin - qmax) * random.random()
            v[i, :] = v[i, :] + (Sol[i, :] - best) * q[i]
            S[i, :] = Sol[i, :] + v[i, :]

            # Check boundaries
            Sol = numpy.clip(Sol, lb, ub)

            # Pulse rate
            if random.random() > r:
                S[i, :] = best + 0.001 * numpy.random.randn(dimension)

            # Evaluate new solutions
            Fnew = objf(S[i, :])

            # Update if the solution improves
            if ((Fnew <= Fitness[i]) and (random.random() < a)):
                Sol[i, :] = numpy.copy(S[i, :])
                Fitness[i] = Fnew

            # Update the current best solution
            if Fnew <= fmin:
                best = S[i, :]
                fmin = Fnew

        # update convergence curve
        Convergence_curve.append(fmin)

        if (t % 1 == 0):
            print(['At iteration ' + str(t) + ' the best fitness is ' + str(fmin)])

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "BAT"
    s.objfname = objf.__name__

    return s
