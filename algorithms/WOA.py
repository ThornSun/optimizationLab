import numpy as np


def woa(maxiteration, population, demension, fitness):
    x = np.random.randint(2, size=(population,demension))
