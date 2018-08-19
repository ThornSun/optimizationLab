import math
import numpy as np


def lapo(maxiteration, population, demension, fitness):
    global result
    x = np.random.randint(2, size=(population, demension))
    new_x = np.random.randint(2, size=(population, demension))
    fx = np.zeros(population)
    new_fx = np.zeros(population)
    cg_curve = np.zeros(maxiteration)
    xavg = np.mean(x, axis=0)
    # xavg二进制化
    favg = fitness(xavg)
    for i in range(population):
        fx[i] = fitness(x[i, :])
    for iteration in range(maxiteration):
        # initialization将数据集中适应度小于平均值的数据用平均值代替
        result = sorted(enumerate(fx), key=lambda index: index[1])
        b = [index[0] for index in result]
        if favg > fx[b[1], 1]:
            fx[b[1], 1] = favg
            x[b[1], 1] = xavg
        xavg = np.mean(x, axis=0)
        favg = fitness(xavg)
        # next junmp determination
        for i in range(population):
            j = np.random.randint(1, population)
            if favg > fx[j]:
                for k in range(demension):
                    new_x[i, k] = x[i, k] + np.random.random() * (xavg[k] - np.random.random() * x[j, k])
            else:
                for k in range(demension):
                    new_x[i, k] = x[i, k] - np.random.random() * (xavg[k] - np.random.random() * x[j, k])
            new_fx[i] = fitness(new_x[i, :])
            # branch fading
            if new_fx[i] < fx[i]:
                new_fx[i] = fx[i]
                new_x[i, :] = x[i, :]
        # upwardMovement
        fx = new_fx
        x = new_x
        result = sorted(enumerate(fx), key=lambda index: index[1])
        b = [index[0] for index in result]
        for i in range(population):
            for j in range(demension):
                s = 1 - (iteration * 1 / maxiteration) * math.exp(-iteration * 1 / maxiteration)
                new_x[i, j] = x[i, j] + np.random.random() * s * (x[b[population], j] - x[b[1], j])
            new_fx[i] = fitness(new_x[i, :])
            # branch fading
            if new_fx[i] < fx[i]:
                new_fx[i] = fx[i]
                new_x[i, :] = x[i, :]
        fx = new_fx
        x = new_x
        result = sorted(enumerate(fx), key=lambda index: index[1])
        cg_curve[iteration] = [index[1] for index in result]
    a = [index[1] for index in result]
    b = [index[0] for index in result]
    best_pos = x[b[population], :]
    best_score = a[population]
    return best_pos, best_score, cg_curve
