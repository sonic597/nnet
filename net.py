import random
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import dataprep.irisdata as DATA # import data, parameters
import matplotlib.colors as colors

layout = DATA.layout # the layers excluding bias units
xs = DATA.x 
ys = DATA.y
rp = DATA.rp #range of random values in the initial thetas
lrate = DATA.lrate # learning rate
iters = DATA.iters # iterations over training data
reg = DATA.reg #regularisation coefficient
feature_names = DATA.fnames #names of features
output_names = DATA.onames #names of output

arch = [x+1 for x in layout[:-1]] + [layout[-1]] #the layers including bias units
activations = [[0]*i for i in arch] # list for neurons, all set to 0

def t(rrange): #initalise theta matrix
    q =[]
    for layer in range(1, len(layout)):
        _l = []
        for neuron in range(layout[layer]):
            _l.append([random.random()*2*rrange-rrange for i in range(arch[layer-1])])
        q.append(_l)
    return q

thetas = t(rp) #random range of theta values
dcounter = t(0) # to add up new theta values amongst iterations. list with same dimensions as thetas but all 0's


def dp(x,y): #dotprod of 2 vectors
    return sum([a*b for a,b in zip(x,y)])

def sigmoid(x): # maps real number to => (0,1)
    return 1/(1+math.exp(-1*x))

def cost(y): #needs to summed, divided by data size
    c = 0
    for output in range(len(y)):
        if y[output] == 1:
            c+= math.log(activations[-1][output])
        else:
            c+= math.log(1-activations[-1][output])
    return -c

def forward(x): #some vector training example x
    activations[0] = [1]+x
    for i in range(len(activations)-1): #all bias units activations = 1
        activations[i][0] = 1
    for layer in range(1, len(layout)): #iter thru each layer
        for neuron in range(layout[layer]): #iter thru units excl bias in layer
            if layer == len(layout) - 1: #last layer has no bias unit
                activations[layer][neuron] = sigmoid(dp(thetas[layer-1][neuron], activations[layer-1]))
            else:
                activations[layer][neuron+1] = sigmoid(dp(thetas[layer-1][neuron], activations[layer-1]))
    _w = activations[-1].copy() #return the output layer as the hypothesis, copy to avoid broadcasting
    return _w
def backward(y): #some vector correct output y
    deltas = [[0]*i for i in layout] # deltas for each neuron
    m_act = [i[1:] for i in activations[:-1]] + [activations[-1]] #modified activations list w/o bis units
    deltas[-1] = [m_act[-1][i]-y[i] for i in range(len(y))]
    #iter back from last layer to the 2nd hidden layer (layer index 2)
    for layer in range(len(layout)-1, 1, -1):
        for n1 in range(layout[layer-1]): # the delta neuron
            ds = m_act[layer-1][n1]*(1-m_act[layer-1][n1]) #derivative of sigmoid 
            for n2 in range(layout[layer]): # the prev. neuron w/ known delta val
                deltas[layer-1][n1] += deltas[layer][n2]*thetas[layer-1][n2][n1+1]*ds #[n1+1] because theta matrix incl. bias
    #updating dcounter
    for L in range(len(dcounter)):#iter by layer thru dcounter [L]
        for N in range(len(dcounter[L])):# iter by neuron [N]
            for J in range(len(dcounter[L][N])):# iter through each connection [J]
                dcounter[L][N][J] += activations[L][J]*deltas[L+1][N] #activation of current neuron*error on reciving neuron

data_size = len(xs)
costs = [] #list of costs over iterations
for iteration in range(iters): #iterate
    err = 0 #error/cost
    for datapoint in range(len(xs)):
        forward(xs[datapoint])
        backward(ys[datapoint])
        err += cost(ys[datapoint])
    costs.append(err/data_size)
    # perform gradient descent
    for L in range(len(dcounter)):#iter by layer thru dcounter [L]
        for N in range(len(dcounter[L])):# iter by neuron [N]
            for J in range(len(dcounter[L][N])):# iter through each connection [J]
                #regularisation:
                if J == 0:
                    _l = dcounter[L][N][J]
                else:
                    _l = dcounter[L][N][J] + reg*thetas[L][N][J]
                thetas[L][N][J] -= lrate/data_size * _l

'''
# Used to observe effect of regularisation. "s" is the sum of the absolute values of all thetas.
# Stronger regularisation should result in smaller s
s = 0
for L in thetas:
        for N in L:
            for J in N:
                s += abs(J)
'''
mins = [q for q in xs[0]]
maxs = [q for q in xs[0]]

for datapoint in xs: # getting min and max of each feature
    for feature in range(len(datapoint)):
        if datapoint[feature] > maxs[feature]:
            maxs[feature] = datapoint[feature]
        elif datapoint[feature] < mins[feature]:
            mins[feature] = datapoint[feature]

fig, ax = plt.subplots()
ax.plot(costs)
ax.set_xlabel("Iterations")
ax.set_ylabel("Cost")
ax.set_title("Waiting...")

divider_line = 0.6 #the spliting point between graph and sliders [0,1]

plt.subplots_adjust(right=divider_line, left = 0.1)


def update(val): # let users put their own inputs
    rawforward = forward([s.val for s in sliders])
    m = max(rawforward)
    _in = rawforward.index(m)
    hypo = output_names[_in] # hypothesis
    colour = colors.hsv_to_rgb([_in/(len(output_names)),0.5,0.5]) # have each output as a distinct colour on the HSV spectrum
    m *= 100 # get "certainty" percentage
    m = round(m, 2)
    tt = str(hypo) + ", " + str(m)+ "%" + " certainty"
    ax.set_title(tt, color=colour)

sliders = []
for i in range(layout[0]):
    slider_ax = plt.axes([divider_line+((1-divider_line)*i/layout[0]),0.1, 0.1, 0.8])
    slider = Slider(slider_ax, feature_names[i], valmin=mins[i], valmax=maxs[i], valinit=(mins[i]+maxs[i])/2, orientation="vertical")
    slider.on_changed(update)
    sliders.append(slider)

inp = [q for q in xs[0]] # the input to hyp

print("FINAL COST: ", costs[-1])
plt.show()
