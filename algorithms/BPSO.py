import numpy
import math


def bpso(maxiteration, population, dimension, objf):
    # initialization
    wmax = 0.9
    wmin = 0.4
    c1 = 2
    c2 = 2
    vmax = 6
    velocity = numpy.zeros((population, dimension))
    position = numpy.random.randint(2, size=(population, dimension))
    pbest = numpy.zeros((population, dimension))
    pbestscore = numpy.zeros(population)
    gbest = numpy.zeros((1, dimension))
    gbestscore = 0
    cg_curve = numpy.zeros((1, maxiteration))
    for iteration in range(maxiteration):
        for i in range(population):
            fx = objf(position[i, :])
            if pbestscore[i] < fx:
                pbestscore[i] = fx
                pbest[i, :] = position[i, :]
            if gbestscore < fx:
                gbestscore = fx
                gbest[0] = position[i, :]
        # update the w of PSO
        w = wmax - iteration * ((wmax - wmin) / maxiteration)
        for i in range(population):
            for j in range(dimension):
                velocity[i, j] = w * velocity[i, j] + c1 * numpy.random.random() * (
                        pbest[i, j] - position[i, j]) + c2 * numpy.random.random() * (gbest[j] - position[i, j])
                if velocity[i, j] > vmax:
                    velocity[i, j] = vmax
                if velocity[i, j] < -vmax:
                    velocity[i, j] = -vmax
                s = math.fabs((2 / math.pi) * math.atan((math.pi / 2) * velocity[i, j]))
                if numpy.random.random() < s:
                    if position[i, j] == 0:
                        position[i, j] = 1
                    else:
                        position[i, j] = 0
        cg_curve[iteration] = gbestscore
    return gbest, gbestscore, cg_curve
