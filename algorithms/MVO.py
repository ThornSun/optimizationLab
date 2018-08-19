import random
import numpy
import time
import math
from numpy import asarray
from sklearn.preprocessing import normalize
from Solution import Solution


def normr(Mat):
    """normalize the columns of the matrix
    B= normr(A) normalizes the row
    the dtype of A is float"""
    Mat = Mat.reshape(1, -1)
    # Enforce dtype float
    if Mat.dtype != 'float':
        Mat = asarray(Mat, dtype=float)

    # if statement to enforce dtype float
    B = normalize(Mat, norm='l2', axis=1)
    B = numpy.reshape(B, -1)
    return B


def randk(t):
    if (t % 2) == 0:
        s = 0.25
    else:
        s = 0.75
    return s


def RouletteWheelSelection(weights):
    accumulation = numpy.cumsum(weights)
    p = random.random() * accumulation[-1]
    chosen_index = -1
    for index in range(0, len(accumulation)):
        if (accumulation[index] > p):
            chosen_index = index
            break

    choice = chosen_index

    return choice


def mvo(maxiteration, population, dimension, objf, lb, ub):
    "parameters"
    # dim=30
    # lb=-100
    # ub=100
    WEP_Max = 1
    WEP_Min = 0.2
    # Max_time=1000
    # N=50

    Universes = numpy.random.uniform(0, 1, (population, dimension)) * (ub - lb) + lb
    Sorted_universes = numpy.copy(Universes)

    convergence = numpy.zeros(maxiteration)

    Best_universe = [0] * dimension
    Best_universe_Inflation_rate = float("inf")

    s = Solution()

    Time = 1
    ############################################
    print("MVO is optimizing  \"" + objf.__name__ + "\"")

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    while (Time < maxiteration + 1):

        "Eq. (3.3) in the paper"
        WEP = WEP_Min + Time * ((WEP_Max - WEP_Min) / maxiteration)

        TDR = 1 - (math.pow(Time, 1 / 6) / math.pow(maxiteration, 1 / 6))

        Inflation_rates = [0] * len(Universes)

        for i in range(0, population):
            Universes[i, :] = numpy.clip(Universes[i, :], lb, ub)

            Inflation_rates[i] = objf(Universes[i, :])

            if Inflation_rates[i] < Best_universe_Inflation_rate:
                Best_universe_Inflation_rate = Inflation_rates[i]
                Best_universe = numpy.array(Universes[i, :])

        sorted_Inflation_rates = numpy.sort(Inflation_rates)
        sorted_indexes = numpy.argsort(Inflation_rates)

        for newindex in range(0, population):
            Sorted_universes[newindex, :] = numpy.array(Universes[sorted_indexes[newindex], :])

        normalized_sorted_Inflation_rates = numpy.copy(normr(sorted_Inflation_rates))

        Universes[0, :] = numpy.array(Sorted_universes[0, :])

        for i in range(1, population):
            Back_hole_index = i
            for j in range(0, dimension):
                r1 = random.random()

                if r1 < normalized_sorted_Inflation_rates[i]:
                    White_hole_index = RouletteWheelSelection(-sorted_Inflation_rates)

                    if White_hole_index == -1:
                        White_hole_index = 0
                    White_hole_index = 0
                    Universes[Back_hole_index, j] = Sorted_universes[White_hole_index, j]

                r2 = random.random()

                if r2 < WEP:
                    r3 = random.random()
                    if r3 < 0.5:
                        Universes[i, j] = Best_universe[j] + TDR * (
                                    (ub - lb) * random.random() + lb)  # random.uniform(0,1)+lb)
                    if r3 > 0.5:
                        Universes[i, j] = Best_universe[j] - TDR * (
                                    (ub - lb) * random.random() + lb)  # random.uniform(0,1)+lb)

        convergence[Time - 1] = Best_universe_Inflation_rate
        if (Time % 1 == 0):
            print(['At iteration ' + str(Time) + ' the best fitness is ' + str(Best_universe_Inflation_rate)])

        Time = Time + 1
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence
    s.optimizer = "MVO"
    s.objfname = objf.__name__

    return s
