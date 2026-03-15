import random 

# Addressed issues 
'confirmed changes as of now'
# Neuron object will have a constructor that only takes bias and weights as part of the object 
# inputs and outputs will be parts of the process , not the object
# Additional Network class is created to convert it into a full fleged network 
# functions like backprop will be present in both neuron and network classes , in network to direct to right neuron and in neuron to change the actual weights and biases 
# additionally , Instead of doing weighted multiplication of next layer , we do it for the inputs of previous layers
# we also add a Layer class before adding a Network class for modularity and better data flow understanding 
# also the type of layer , input | hidden | output should not be the concern of the neuron class but the Layer class


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# New issues 
# Security(memory) and input validation related bugs and edge case bugs not looked after
# Assumption that Layer and Neuron or maybe something totally different (not built/designed yet) will handle activation and training (proper architecture not made yet)
# this is just one indevidual neuron that does not yet implement activation in itself , nothing more than that 
# Maybe placing activation functions in layer class is benificial overall (still uncertain)
class Neuron:

    # weights(of connections to previaus layer) and bias(of the neuron) are initialized at the instance creation
    def __init__(self , num_inputs : int ):
        self.bias = random.random()

        self.weights = []

        for i in range(num_inputs):
            self.weights.append(random.random())
        
    # Processing of the incoming data from the neuron 
    def forward_processing(self,inputs : list): 
       
        # calculating the weighted product of inputs and adding the neuron bias to produce the neuron's linear output 
        weighted_input_sum = 0

        for i in range(len(inputs)):
            weighted_input_sum += self.weights[i]*inputs[i]

        weighted_input_sum += self.bias

        return weighted_input_sum # linear output of a neuron
        
        
        
        
    # defining ReLU activation function to be used later in layers class
    def activation_fn_relu(self,input):

        if input <= 0:
            input = 0 
            
        return input
    
    # defining Sigmoid activation function to be used later in layers class
    def activation_fn_sigmoid(self,input):

        
        input = 1/(1+(2.718281828459045**-input))
        
        return input
    
    





    

