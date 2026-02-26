import random


# Fix thought process
# instead of treating input layer as a group of neurons , we treat it as a list of inputs that go into each neuron
# non linearities handled in Layer class
# Input length validation with respect to weights added




'''
Problems - 
   1) The structure is valid but only in the context of forward propogation
   2) there is no learning/training or backpropogation possible with the current structure in itself
   3) Range of the randomly initialized variables (weights and biases) may not be efficient
   4) Activation function choice handling might be needed to be looked at
   5) Error handling need to be looked at and input validation still incomplete , it
   - only validates input length

'''
class Neuron:
    def __init__(self , num_inputs: int):

        self.bias = random.random()
        self.weights = []
        for _ in range(num_inputs):
            self.weights.append(random.random())
    

    def forward_prop_neuron(self, inputs : list):
        output = 0
        if (len(inputs) ==len(self.weights)):
            for i in range(len(inputs)):
                output += inputs[i]*self.weights[i]
            
            output +=self.bias

            return output
    
    
# class layer - takes the number of neurons in current layers and number of neurons in previous layer in constructor 
# number of neurons in current layers help create the same number of neuron instances and store them in a list 
# number of neurons in previous layer helps initialize the number of weights for each corresponding neuron of the current layer 

# Addition - We integrate the activation function to forward prop layer function and add activation function as a part of the Layer object 
class Layer:
    def __init__(self, num_previos_layer_neurons :int , num_current_layer_neurons:int , activation = 'relu'): 

        self.current = []
        self.activation = activation

        for _ in range(num_current_layer_neurons):
            self.current.append(Neuron(num_previos_layer_neurons))
        
    # forward prop layer carries out forward propogation across each neuron in the current layer and store the outputs in a list and returns it
    def forward_prop_layer(self, inputs:list):
        outputs = []
        for neuron in self.current:
            # performs forward prop for each indevidual neuron and stores their output to use as inputs to next layer
            outputs.append(neuron.forward_prop_neuron(inputs))
 
        # integrating non linear functions into forward prop 
        if self.activation == 'relu':
            non_linear_outputs = self.non_linear_relu(outputs)

        elif self.activation == 'sigmoid':
            non_linear_outputs = self.non_linear_sigmoid(outputs)

        else:
            print("Activation function not added")
            exit()
        # returning non linear outputs 
        return non_linear_outputs
    
    def non_linear_relu(self , inputs : list):
        nl_outputs = []
        for i in range(len(inputs)):
            # using seperate list for outputs
            if inputs[i] <= 0:
                nl_outputs.append(0)
            else:
                nl_outputs.append(inputs[i])
        
        return nl_outputs
    
    def non_linear_sigmoid(self , inputs : list):
        nl_outputs = []
        for i in range(len(inputs)):

            # using seperate list for outputs
            nl_outputs.append(1/(1+2.71828**(-inputs[i])))
        
        return nl_outputs

        


        
    






