import numpy as np


class Layer: 
    # NOT exactly a layer with respect to the traditional definition of a layer in neural networks , but more of the Area Between 2 Layers 
    # (inputs to previous layer) Inputs -> Processing (Mathematics) -> Outputs (inputs to next layer) 

    # Defining valid Activations across all instances via shared variable
    valid_activations = ('relu' , 'sigmoid' , 'softmax' , None)
    def __init__(self, input_dim, output_dim , activation = None ):
        self.activation_fn = activation
        self.initialization(in_size=input_dim , out_size=output_dim)


 
    def validation(self):

        if self.activation_fn.lower() not in Layer.valid_activations:
            raise Exception("Not a Valid Activation Function")
        
    def initialization(self, in_size , out_size): # initializing weight and biases 
        # Conditionally define weight matrix based on input-output sizes and weight matrix
        # self.activation_fn (working with this ) 
        self.weights = 0
        self.bias = 0
    
    def forward(self, inputs):
        self.validation()
        # Performs forward pass through layer
        self.inputs = inputs
        linear_outs = np.dot(self.inputs, self.weights) + self.bias
        self.forward_outs = self.activation(linear = linear_outs)
        return 
    
    def activation(self , linear):
        # Conditionally define activation for forward prop
        pass 

