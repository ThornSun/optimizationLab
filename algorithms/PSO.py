import random
import numpy
from Solution import Solution
import time


def pso(maxiteration, population, dimension, objf, lb, ub):
    # PSO parameters

    #    dim=30
    #    iters=200
    Vmax = 6
    #    PopSize=50     #population size
    wMax = 0.9
    wMin = 0.2
    c1 = 2
    c2 = 2
    #    lb=-10
    #    ub=10
    #
    s = Solution()

    ######################## Initializations

    vel = numpy.zeros((population, dimension))

    pBestScore = numpy.zeros(population)
    pBestScore.fill(float("inf"))

    pBest = numpy.zeros((population, dimension))
    gBest = numpy.zeros(dimension)

    gBestScore = float("inf")

    pos = numpy.random.uniform(0, 1, (population, dimension)) * (ub - lb) + lb

    convergence_curve = numpy.zeros(maxiteration)

    ############################################
    print("PSO is optimizing  \"" + objf.__name__ + "\"")

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(0, maxiteration):
        for i in range(0, population):
            # pos[i,:]=checkBounds(pos[i,:],lb,ub)
            pos[i, :] = numpy.clip(pos[i, :], lb, ub)
            # Calculate objective function for each particle
            fitness = objf(pos[i, :])

            if (pBestScore[i] > fitness):
                pBestScore[i] = fitness
                pBest[i, :] = pos[i, :]

            if (gBestScore > fitness):
                gBestScore = fitness
                gBest = pos[i, :]

        # Update the W of PSO
        w = wMax - l * ((wMax - wMin) / maxiteration)

        for i in range(0, population):
            for j in range(0, dimension):
                r1 = random.random()
                r2 = random.random()
                vel[i, j] = w * vel[i, j] + c1 * r1 * (pBest[i, j] - pos[i, j]) + c2 * r2 * (gBest[j] - pos[i, j])

                if (vel[i, j] > Vmax):
                    vel[i, j] = Vmax

                if (vel[i, j] < -Vmax):
                    vel[i, j] = -Vmax

                pos[i, j] = pos[i, j] + vel[i, j]

        convergence_curve[l] = gBestScore

        if (l % 1 == 0):
            print(['At iteration ' + str(l + 1) + ' the best fitness is ' + str(gBestScore)])
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "PSO"
    s.objfname = objf.__name__

    return s
