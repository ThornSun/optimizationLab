import numpy
from Solution import Solution
import time


def gwo(maxiteration, population, dimension, objf, lb, ub):
    # initialize alpha, beta, and delta_pos
    Alpha_pos = numpy.zeros(dimension)
    Alpha_score = float("inf")

    Beta_pos = numpy.zeros(dimension)
    Beta_score = float("inf")

    Delta_pos = numpy.zeros(dimension)
    Delta_score = float("inf")

    # Initialize the positions of search agents
    Positions = numpy.random.uniform(0, 1, (population, dimension)) * (ub - lb) + lb

    Convergence_curve = numpy.zeros(maxiteration)
    s = Solution()

    # Loop counter
    print("GWO is optimizing  \"" + objf.__name__ + "\"")

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    for l in range(0, maxiteration):
        for i in range(0, population):

            # Return back the search agents that go beyond the boundaries of the search space
            Positions[i, :] = numpy.clip(Positions[i, :], lb, ub)

            # Calculate objective function for each search agent
            fitness = objf(Positions[i, :])

            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Alpha_score = fitness  # Update alpha
                Alpha_pos = Positions[i, :]

            if (fitness > Alpha_score and fitness < Beta_score):
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :]

            if (fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score):
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :]

        a = 2 - l * ((2) / maxiteration)  # a decreases linearly fron 2 to 0

        # Update the Position of search agents including omegas
        for i in range(0, population):
            for j in range(0, dimension):
                r1 = numpy.random.random()  # r1 is a random number in [0,1]
                r2 = numpy.random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a  # Equation (3.3)
                C1 = 2 * r2  # Equation (3.4)

                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])  # Equation (3.5)-part 1
                X1 = Alpha_pos[j] - A1 * D_alpha  # Equation (3.6)-part 1

                r1 = numpy.random.random()
                r2 = numpy.random.random()

                A2 = 2 * a * r1 - a  # Equation (3.3)
                C2 = 2 * r2  # Equation (3.4)

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])  # Equation (3.5)-part 2
                X2 = Beta_pos[j] - A2 * D_beta  # Equation (3.6)-part 2

                r1 = numpy.random.random()
                r2 = numpy.random.random()

                A3 = 2 * a * r1 - a  # Equation (3.3)
                C3 = 2 * r2  # Equation (3.4)

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])  # Equation (3.5)-part 3
                X3 = Delta_pos[j] - A3 * D_delta  # Equation (3.5)-part 3

                Positions[i, j] = (X1 + X2 + X3) / 3  # Equation (3.7)

        Convergence_curve[l] = Alpha_score

        if (l % 1 == 0):
            print(['At iteration ' + str(l) + ' the best fitness is ' + str(Alpha_score)])

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "GWO"
    s.objfname = objf.__name__

    return s
