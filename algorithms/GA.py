import numpy as np


def ga(maxiteration, population, demension, fitness):
    crossRate = 0.8
    mutationRate = 0.1
    x = np.random.randint(2, size=(population,demension))
