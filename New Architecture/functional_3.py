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
        if self.inputs.shape[0] != self.weights.shape[0]:
            raise Exception("Data Shape Not same as expected Input shape")
        
        
    def initialization(self, in_size , out_size): # initializing weight and biases 
        # Conditionally define weight matrix based on input-output sizes and weight matrix
        # self.activation_fn (working with this ) 
        self.weights = 0
        self.bias = 0
    
    def forward(self, inputs:np.typing.NDArray):
        self.inputs = inputs
        self.validation()
        # Performs forward pass through layer
        linear_outs = np.dot(self.inputs, self.weights) + self.bias # Linear pass 
        self.forward_outs = self.activation(linear = linear_outs) # Activation
        return self.forward_outs # Forwarded to next layer as inputs
    
    def activation(self , linear):

        Activated_Outputs = None
        match self.activation_fn.lower():
            
            case 'sigmoid':
                Activated_Outputs = 1 / (1 + (np.exp(-linear))) 

            case 'relu':
                Activated_Outputs = np.maximum(0,linear)

            case 'softmax':
                # For NOT Computationally destroying anything
                shift = linear - max(linear) 
                exponents = np.exp(shift)
                summation = np.sum(exponents)
                Activated_Outputs = exponents/summation
            
            case None:
                Activated_Outputs = linear

        if Activated_Outputs == None:
            raise Exception("Activation Failed")

        return Activated_Outputs
    
    # Forward pass Done for the layer

    def backprop(self,non_linear_loss):
        pass

    def linearization(self,non_linear_loss):
        pass
    
    

