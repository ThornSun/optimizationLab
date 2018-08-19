def binarization(initialization):
    row = initialization.shape[0]
    column = initialization.shape[1]
    for i in range(row):
        for j in range(column):
            if initialization[i,j]<0.5:
                initialization[i,j] = 0;
            else:
                initialization[i,j] = 1;
    return initialization
