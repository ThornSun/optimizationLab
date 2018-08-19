import numpy
import math
import time
from Solution import Solution


def alpha_new(alpha, NGen):
    # % alpha_n=alpha_0(1-delta)^NGen=10^(-4)
    # % alpha_0=0.9
    delta = 1 - (10 ** (-4) / 0.9) ** (1 / NGen)
    alpha = (1 - delta) * alpha
    return alpha


def ffa(maxiteration, population, dimension, objf, lb, ub):
    # FFA parameters
    alpha = 0.5  # Randomness 0--1 (highly random)
    betamin = 0.20  # minimum value of beta
    gamma = 1  # Absorption coefficient

    zn = numpy.ones(population)
    zn.fill(float("inf"))

    # ns(i,:)=Lb+(Ub-Lb).*rand(1,d)
    ns = numpy.random.uniform(0, 1, (population, dimension)) * (ub - lb) + lb
    Lightn = numpy.ones(population)
    Lightn.fill(float("inf"))

    # [ns,Lightn]=init_ffa(n,d,Lb,Ub,u0)

    convergence = []
    s = Solution()

    print("CS is optimizing  \"" + objf.__name__ + "\"")

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main loop
    for k in range(0, maxiteration):  # start iterations

        # % This line of reducing alpha is optional
        alpha = alpha_new(alpha, maxiteration)

        # % Evaluate new solutions (for all n fireflies)
        for i in range(0, population):
            zn[i] = objf(ns[i, :])
            Lightn[i] = zn[i]

        # Ranking fireflies by their light intensity/objectives

        Lightn = numpy.sort(zn)
        Index = numpy.argsort(zn)
        ns = ns[Index, :]

        # Find the current best
        nso = ns
        Lighto = Lightn
        nbest = ns[0, :]
        Lightbest = Lightn[0]

        # % For output only
        fbest = Lightbest

        # % Move all fireflies to the better locations
        #    [ns]=ffa_move(n,d,ns,Lightn,nso,Lighto,nbest,...
        #          Lightbest,alpha,betamin,gamma,Lb,Ub)
        scale = numpy.ones(dimension) * abs(ub - lb)
        for i in range(0, population):
            # The attractiveness parameter beta=exp(-gamma*r)
            for j in range(0, population):
                r = numpy.sqrt(numpy.sum((ns[i, :] - ns[j, :]) ** 2))
                # r=1
                # Update moves
                if Lightn[i] > Lighto[j]:  # Brighter and more attractive
                    beta0 = 1
                    beta = (beta0 - betamin) * math.exp(-gamma * r ** 2) + betamin
                    tmpf = alpha * (numpy.random.rand(dimension) - 0.5) * scale
                    ns[i, :] = ns[i, :] * (1 - beta) + nso[j, :] * beta + tmpf

        # ns=numpy.clip(ns, lb, ub)

        convergence.append(fbest)

        IterationNumber = k
        BestQuality = fbest

        if (k % 1 == 0):
            print(['At iteration ' + str(k) + ' the best fitness is ' + str(BestQuality)])
    #    
    ####################### End main loop
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence
    s.optimizer = "FFA"
    s.objfname = objf.__name__

    return s
