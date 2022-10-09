import csv
import random
# Iris-setosa = [1,0,0]
# Iris-versicolor = [0,1,0]
# Iris-virginica = [0,0,1]
x = []
y = []
rp = 1
layout = [4,6,3] 
lrate = 0.3
iters = 500
reg = 100
fnames = ["s_length", "s_width", "p_length", "p_width"]
onames = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

with open("dataprep/iris.csv") as dt: # name file as relative path from network
    c = list(csv.reader(dt))
    for dp in c:
        xl = [float(g) for g in dp[:-1]]
        _i = dp[-1]
        if _i == "Iris-setosa":
            yl = [1,0,0]
        elif _i == "Iris-versicolor":
            yl = [0,1,0]
        elif _i == "Iris-virginica":
            yl = [0,0,1]
        rr = random.randint(0,len(x)) # shuffle the lists
        x.insert(rr, xl)
        y.insert(rr, yl)
