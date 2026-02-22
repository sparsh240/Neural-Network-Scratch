import numpy as np

MIN_INIT = -0.4
MAX_INIT =  0.4

# Network class added for a complete forward pass handling 
# Numpy added for better mathematical calculations and faster calculations 

'''
Still not ready for backprop or training
Basic network structure , not resembling modern neural networks Yet
relatively Slow
MAJOR ISSUES - 
Giving ownership of weights to neurons causes problems with storage in forward or backward pass
It is not feasible for one of forward or backward pass to take place without consuming too much time or storage 
overall very hectic to let a neuron or layer control specific weights
'''
class Neuron:

    def __init__(self , input_size : int):
        # Defining weights and biases
        self.bias = 0
        self.weights = np.random.uniform(MIN_INIT , MAX_INIT , input_size)

    def input_validation(self, inputs : np.typing.NDArray):
        # Checking input size integrity 
        if self.weights.shape != inputs.shape or inputs.dtype not in [ np.float64 , np.float32] :
            raise Exception("Invalid Input Dimensions / Invalid input type")

    def forward(self , inputs : np.typing.NDArray):
        # Required check before proceeding 
        self.input_validation(inputs)

        # Output of the neuron 
        linear_output = np.dot(inputs , self.weights)
        linear_output += self.bias

        return linear_output

    # Defining a backprop function 
    def backward(self):
        pass


class Layer:
    def __init__(self , input_per_neuron : int , current_layer_size : int  , activation_fn : str):
        # Verifying and defining activation function
        self.verify_activation(activation_fn)
        self.activation = activation_fn

        # Defining inputs and outputs sizes to check for connection validity in Network class
        self.num_inputs = input_per_neuron
        self.num_outputs = current_layer_size

        # Creating a layer with independent Neurons 
        self.current_layer = np.array([Neuron(input_per_neuron) for _ in range(current_layer_size)]) # list definition enclosed in np array

    def verify_activation(self , activation : str):
        # Storing the activation functions we generally use
        activation_fns = ['relu' , 'softmax' , 'sigmoid']
        # Checking
        if activation not in activation_fns:
            raise Exception("Invalid Activation Function")



    def forward(self , inputs :np.typing.NDArray):
        outputs = []
        for neuron in self.current_layer:
            output = neuron.forward(inputs)
            outputs.append(output)
        outputs = np.array(outputs)
        final_output = self.non_linear_activation(outputs)

        return final_output
    
    def non_linear_activation(self , outputs : np.typing.NDArray):

        # Code to calculate non linear output with respect to activation finction
        match self.activation.lower():
            case 'sigmoid':
                outputs = 1 / (1 + (np.exp(-outputs)))
            
            case 'relu':
                outputs = np.maximum(0, outputs)
                
            
            case 'softmax':
                'Numerically dengerous as exponent increases fast'
                # summation = np.exp(outputs).sum() 
                # outputs = np.exp(outputs) / summation

                # Instead we shrink the values down so very large values are treated as close to 1 and small values are close to 0
                # 1. Find the max for stability np.max 
                # 2. Shift the values to avoid overflow
                shifted_outputs = outputs - np.max(outputs)
                # 3. Calculate exponents safely
                exp_values = np.exp(shifted_outputs)
                # 4. Calculate probabilities 
                summation = np.sum(exp_values)
                outputs = exp_values / summation

        return outputs
    # Defining a backprop for class 
    def backprop(self):
        pass



# Creating a network class to handle all the layers and process
class Network:

    def __init__(self):
        # Defining a list of layers , this will help us to detemine input layer and help in forward pass
        self.layers = []

    # Function to add a layer to the network
    # inputs_num is the number of inputs given to each neuron of current layer 
    # outputs_num is the number of outputs the current layer gives
    def layer(self , inputs_num : int , outputs_num : int , activation : str):
        # a list of layers 
        self.layers.append(Layer(inputs_num , outputs_num , activation))

    def connectivity_validation(self):
        for i in range(len(self.layers) - 1):
            if (self.layers[i].num_outputs != self.layers[i+1].num_inputs):
                raise Exception(f"Output-Input Mismatch : Layer {i+1}")
    
    def forward(self , network_inputs : np.typing.NDArray):

        # Validating network layer connectivity
        self.connectivity_validation()

        
        # creating a transition variable that translates output of current layer as input to next layer
        outputs = network_inputs
        for i in range(len(self.layers)):
            # Forward pass 
            outputs = self.layers[i].forward(outputs)
        
        return outputs
    


        

            
        
        






        

