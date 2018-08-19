import numpy as np


def gsa(maxiteration, population, demension, fitness):
    x = np.random.randint(2, size=(population,demension))
