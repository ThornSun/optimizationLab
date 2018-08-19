import numpy


def ga(maxiteration, population, dimension, objf, lb, ub):
    crossRate = 0.8
    mutationRate = 0.1
    x = numpy.random.randint(2, size=(population,dimension))
