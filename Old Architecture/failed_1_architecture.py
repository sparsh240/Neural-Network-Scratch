import random 


# issue - initial layer will not have any weights installed - another class for that needed
# issue - there is no non linear activation function - another class for that also needed , or another internal function 
# OOP's issue - the constructor is treating inputs as parts of the object itself , when it is just valies passing onto the object

class Neuron: 
   
    def __init__(self, inputs_previous_layer, num_outputs):

        # bias of the neuron 
        self.bias = random.random()

        # creating a list to store all weights systematically
        self.__weights = []

        # list to store input of each next neuron 
        

        # inputs to the neuron 
        self.inputs = inputs_previous_layer
        

        # definig 
        for i in range(num_outputs):

            # indevidual weight of each connection 
            self.output_weight = random.random()   

            # adding to weight list         
            self.__weights.append(self.output_weight)
    
    def processing(self):
        
        self.outputs = []
        # weighted sum of inputs 
        weighted_sum  = 0
        for i in self.inputs: 
            weighted_sum += i
        
        # linear output of the neuron 
        linear_output = weighted_sum + self.bias

        # inputs for the next layer
        for i in range(len(self.__weights)):

            # creating inputs for next layers and storing them
            self.outputs.append(linear_output * self.__weights[i]) 

        return self.outputs







        





        
        