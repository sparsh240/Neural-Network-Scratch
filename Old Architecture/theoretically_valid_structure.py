import random 
# logically valid for forward prop for a single layer , next goal will be to work on backprop and slightly optimize this structure every next try and introduce a network class




'Issues:'
' 1) Not a fit structure for backprop, make it backprop compatible'
' 2) Network class is required to handle overall network structure formations and implementations'
' 3) Higly unoptimized structure in the context of speed , memory and efficiency , just a logical structure that is theoretically correct'
class Neuron:

    def __init__(self , num_inputs : int):
        
        
        

        MIN_INITIAL = -0.4
        MAX_INITIAL = 0.4

        self.weights = []
        self.bias = 0

        for _ in range(num_inputs):
            self.weights.append(random.uniform(MIN_INITIAL,MAX_INITIAL))

    
    def input_validation(self , inputs : list):

        # all validation code and stopping if invalid
        if (len(self.weights) == len(inputs)):
            pass
        else:
            raise Exception("Invalid length of inputs")


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
        # we will only care for ReLU , Sigmoid and softmax
        valid_activation = ['sigmoid' , 'relu' , 'softmax']

        if (self.activation.lower() in valid_activation):
            IS_VALID = True

        # All validation code ... 

        if(not IS_VALID):
            # Then proceed otherwise stop
            raise Exception("Not a valid activation function")

    

    def activation_fn_implemented_outputs(self , linear_outputs : list):
        non_linear_outputs = []
        # Code to calculate non linear output with respect to activation finction
        if self.activation.lower() == 'sigmoid':
            for linear_output in linear_outputs:

                linear_output = 1/(1+2.7182**(-linear_output))

                non_linear_outputs.append(linear_output)
        
        elif self.activation.lower() == 'relu':
            for linear_output in linear_outputs:

                if linear_output < 0 :
                    linear_output = 0
                
                non_linear_outputs.append(linear_output)
        
        elif self.activation.lower() == 'softmax':

            summation = 0 
            for linear_output in linear_outputs:
                summation += 2.7182**linear_output


            for linear_output in linear_outputs:
                linear_output = (2.7182**linear_output)/summation

                non_linear_outputs.append(linear_output)


        return non_linear_outputs



    def forward_prop_per_layer(self , inputs : list):

        layer_linear_outputs = []

        for neuron in self.working_layer:
            liner_output = neuron.forward_prop_per_neuron(inputs)

            layer_linear_outputs.append(liner_output)
        
        # integration of non linear outputs into forward propogation 
        final_outputs = self.activation_fn_implemented_outputs(layer_linear_outputs)
        
        return final_outputs
    





    






        
