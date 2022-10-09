# test data outputs 1 iff x1 > 0.5 AND x2 > 0.5
layout = [2,1] # the layers excluding bias units
x = [[0.2,0.7], [0.1,0.4], [0.67,0.3], [0.8,0.9]] #learn true if 2 ones (AND)
y = [[0],[0],[0],[1]]
rp = 1 #range of random values in the initial thetas
lrate = 0.3 # learning rate
iters = 100 # iterations over training data
reg = 1
fnames = ["x1", "x2"]
onames = ["TRUE"]