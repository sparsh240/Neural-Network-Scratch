'''  New architecture-
     1) we treat the areas between the layers as black boxes that perform all the mathematics on the weights and biases
        independent of the ownership of indevidual neurons ,instead of using Arrays inside arrays or 1D arrays , we will be using matrices , now connections will be worked with INDEPENDENTLY
        no layer or neuron needs to work with a specific set of weights , the series of weights used in calculation May vary depending on forward / backward pass

     2) Neurons will NOT be treated as objects with this architecture  , hence we remove the neuron class completely , 
        neurons would rather be related to the row/column of the matrix that carries the weights connected 
        to that specific neuron.

     3) Layer class will be handling the flow of data from one Blackbox to next Blackbox , directing data  where it needs to be , Network
        class Handles the creation and outputs of overall network and Output layer loss
        
'''

import numpy

class Blackbox:
   pass

class Layer:
   pass

class Network:
   pass


        
        
    









\


