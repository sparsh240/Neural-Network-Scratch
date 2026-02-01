import random 


# logically , Forward Propogation for a single laayer is totally possible with the current structure (not for the whole network)


class Neuron:

    def __init__(self , num_inputs : int):
        
        
        

        MIN_INITIAL = -0.4
        MAX_INITIAL = 0.4

        self.weights = []
        self.bias = random.uniform(MIN_INITIAL,MAX_INITIAL)

        for _ in range(num_inputs):
            self.weights.append(random.uniform(MIN_INITIAL,MAX_INITIAL))

    
    def input_validation(self , inputs : list):

        # all validation code and stopping if invalid
        pass

    # Taking Output from previous layer and calculating weighted sum for previous layer and then adding bias
    def forward_prop_per_neuron(self,inputs : list):

        self.input_validation(inputs)

        linear_output = 0

        for i in range(len(inputs)):
            linear_output += inputs[i] * self.weights[i]
        
        linear_output += self.bias

        return linear_output
    



class Layer:
    # Number of neurons in previous layer and current layer

    def __init__(self , previous_layer_neurons : int ,current_layer_neurons : int , activation_fn : str):

        self.activation = activation_fn
        self.activation_verifier()
        self.working_layer = []

        for _ in range(current_layer_neurons):

            self.working_layer.append(Neuron(previous_layer_neurons))
        

        

        
    
    def activation_verifier(self):

        # initializing the flag as false and checking if true , not proceeding if false
        IS_VALID = False

        # All validation code ... 

        if(IS_VALID):
            # Then proceed otherwise stop
            pass

    

    def activation_fn_implemented_outputs(self , linear_outputs : list):
        non_linear_outputs = []
        # Code to calculate non linear output with respect to activation finction

        return non_linear_outputs



    def forward_prop_per_layer(self , inputs : list):

        layer_linear_outputs = []

        for neuron in self.working_layer:
            liner_output = neuron.forward_prop_per_neuron(inputs)

            layer_linear_outputs.append(liner_output)
        
        # integration of non linear outputs into forward propogation 
        final_outputs = self.activation_fn_implemented_outputs(layer_linear_outputs)
        
        return final_outputs
    

class Network:
    # Code to make forward propogation possible for the Network and not just a single layer
    pass

    






        
