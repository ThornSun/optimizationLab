import math
import numpy


def lapo(maxiteration, population, dimension, objf):
    global result
    x = numpy.random.randint(2, size=(population, dimension))
    new_x = numpy.random.randint(2, size=(population, dimension))
    fx = numpy.zeros(population)
    new_fx = numpy.zeros(population)
    cg_curve = numpy.zeros(maxiteration)
    xavg = numpy.mean(x, axis=0)
    # xavg二进制化
    favg = objf(xavg)
    for i in range(population):
        fx[i] = objf(x[i, :])
    for iteration in range(maxiteration):
        # initialization将数据集中适应度小于平均值的数据用平均值代替
        result = sorted(enumerate(fx), key=lambda index: index[1])
        b = [index[0] for index in result]
        if favg > fx[b[1], 1]:
            fx[b[1], 1] = favg
            x[b[1], 1] = xavg
        xavg = numpy.mean(x, axis=0)
        favg = objf(xavg)
        # next junmp determination
        for i in range(population):
            j = numpy.random.randint(1, population)
            if favg > fx[j]:
                for k in range(dimension):
                    new_x[i, k] = x[i, k] + numpy.random.random() * (xavg[k] - numpy.random.random() * x[j, k])
            else:
                for k in range(dimension):
                    new_x[i, k] = x[i, k] - numpy.random.random() * (xavg[k] - numpy.random.random() * x[j, k])
            new_fx[i] = objf(new_x[i, :])
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
            for j in range(dimension):
                s = 1 - (iteration * 1 / maxiteration) * math.exp(-iteration * 1 / maxiteration)
                new_x[i, j] = x[i, j] + numpy.random.random() * s * (x[b[population], j] - x[b[1], j])
            new_fx[i] = objf(new_x[i, :])
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
