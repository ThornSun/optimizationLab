import numpy


def alo(maxiteration, population, dimension, objf, lb, ub):
    x = numpy.random.randint(2, size=(population,dimension))
