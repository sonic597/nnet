# nnet
A Neural Network implementation using gradient descent in Python (sans NumPy)
## Using nnet
`net.py` - the main Python file
`dataprep` - folder where data can be located
### Loading Data
In `net.py`, one may adjust the imports to laod in a data file. For example, for an example data file called `example.py` (located in the `dataprep` folder), the import statement `import dataprep.irisdata as DATA` should be changed to `import dataprep.example as DATA`. (line 5, `net.py`).
The data for `dataprep/irisdata.py` is actually a CSV file, but the library `csv` is used to process it into the required format.
### Adjustable parameters 
One may adjust all the parameters of the neural network directly from the data files. See `/dataprep/ANDdata.py` for an example.
- **Network architecture:** create a variable `layout` to describe the network architecture as a list, with each element determining the number of neurons in the layer. The 1st entry is the input layer, the last one is the output layer.
- **Training data:** all input values for training data should be in the list `x`, each training example being an list within in the list with as many elements as input neurons. The correct outputs should be lists within list `y` and, with each example having the same length as the number of output nodes.
- **Initial random range of weights:** the variable `rp` determines the range of random inital weights (optional parameter. Setting `rp=1` usually works fine).
- **Learning rate of network:** This can be adjusted by variable `lrate`. 
- **Iterations of neural network:** The variable `iters` determines the number of iterations over the training data done by the network.
- **Regularisation:** Regularisation works to combat overfitting to the data by discouraging complecated hypothesis functions. A high value of `reg` corresponds to more regularisation in the network (ie: simple hypotheses are prefered).
- **Feature names:** List `fnames` is a list of the names of the features (as a string) in the neural network. Must be the same legnth as a single training example in `x`.
- **Output names:** `onames` is a list of the names of the individual outputs in the neural network (see `/dataprep/irisdata.py` for an example). Must be the same length as the number of output neurons.

## Functionality
Upon running the file `net.py` with all data properly configured, the network starts training and after it is finished, a GUI will appear in a new window. The `cost` of the network (the error between predictions and true values) is graphed over the number of iterations allowing you to track the progress of the network. The sliders on the side allow you to adjust different values for the features (as named in `fnames`) and the network's prediction for those values along with the certainty for its prediction is shown above the graph.

## Requirements
- Python 3.7 or above
- Matplotlib (can be installed via `pip`)
