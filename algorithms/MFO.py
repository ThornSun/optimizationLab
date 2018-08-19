import numpy
import math
from Solution import Solution
import time


def mfo(maxiteration, population, dimension, objf, lb, ub):
    # Initialize the positions of moths
    Moth_pos = numpy.random.uniform(0, 1, (population, dimension)) * (ub - lb) + lb
    Moth_fitness = numpy.full(population, float("inf"))
    # Moth_fitness=numpy.fell(float("inf"))

    Convergence_curve = numpy.zeros(maxiteration)

    sorted_population = numpy.copy(Moth_pos)
    fitness_sorted = numpy.zeros(population)
    #####################
    best_flames = numpy.copy(Moth_pos)
    best_flame_fitness = numpy.zeros(population)
    ####################
    double_population = numpy.zeros((2 * population, dimension))
    double_fitness = numpy.zeros(2 * population)

    double_sorted_population = numpy.zeros((2 * population, dimension))
    double_fitness_sorted = numpy.zeros(2 * population)
    #########################
    previous_population = numpy.zeros((population, dimension))
    previous_fitness = numpy.zeros(population)

    s = Solution()

    print("MFO is optimizing  \"" + objf.__name__ + "\"")

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    Iteration = 1

    # Main loop
    while (Iteration < maxiteration + 1):

        # Number of flames Eq. (3.14) in the paper
        Flame_no = round(population - Iteration * ((population - 1) / maxiteration))

        for i in range(0, population):
            # Check if moths go out of the search spaceand bring it back
            Moth_pos[i, :] = numpy.clip(Moth_pos[i, :], lb, ub)

            # evaluate moths
            Moth_fitness[i] = objf(Moth_pos[i, :])

        if Iteration == 1:
            # Sort the first population of moths
            fitness_sorted = numpy.sort(Moth_fitness)
            I = numpy.argsort(Moth_fitness)

            sorted_population = Moth_pos[I, :]

            # Update the flames
            best_flames = sorted_population
            best_flame_fitness = fitness_sorted
        else:
            #
            #        # Sort the moths
            double_population = numpy.concatenate((previous_population, best_flames), axis=0)
            double_fitness = numpy.concatenate((previous_fitness, best_flame_fitness), axis=0)
            #
            double_fitness_sorted = numpy.sort(double_fitness)
            I2 = numpy.argsort(double_fitness)
            #
            #
            for newindex in range(0, 2 * population):
                double_sorted_population[newindex, :] = numpy.array(double_population[I2[newindex], :])

            fitness_sorted = double_fitness_sorted[0:population]
            sorted_population = double_sorted_population[0:population, :]
            #
            #        # Update the flames
            best_flames = sorted_population
            best_flame_fitness = fitness_sorted

        #
        #   # Update the position best flame obtained so far
        Best_flame_score = fitness_sorted[0]
        Best_flame_pos = sorted_population[0, :]
        #
        previous_population = Moth_pos
        previous_fitness = Moth_fitness
        #
        # a linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
        a = -1 + Iteration * ((-1) / maxiteration)

        # Loop counter
        for i in range(0, population):
            #
            for j in range(0, dimension):
                if (i <= Flame_no):  # Update the position of the moth with respect to its corresponsing flame
                    #
                    # D in Eq. (3.13)
                    distance_to_flame = abs(sorted_population[i, j] - Moth_pos[i, j])
                    b = 1
                    t = (a - 1) * numpy.random.random() + 1
                    #
                    #                % Eq. (3.12)
                    Moth_pos[i, j] = distance_to_flame * math.exp(b * t) * math.cos(t * 2 * math.pi) + \
                                     sorted_population[i, j]
                #            end
                #
                if i > Flame_no:  # Upaate the position of the moth with respct to one flame
                    #
                    #                % Eq. (3.13)
                    distance_to_flame = abs(sorted_population[i, j] - Moth_pos[i, j])
                    b = 1
                    t = (a - 1) * numpy.random.random() + 1
                    #
                    #                % Eq. (3.12)
                    Moth_pos[i, j] = distance_to_flame * math.exp(b * t) * math.cos(t * 2 * math.pi) + \
                                     sorted_population[Flame_no, j]

        # Display best fitness along the iteration
        if (Iteration % 1 == 0):
            print(['At iteration ' + str(Iteration) + ' the best fitness is ' + str(Best_flame_score)])

        Iteration = Iteration + 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "MFO"
    s.objfname = objf.__name__

    return s
