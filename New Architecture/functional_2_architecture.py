import numpy as np


# Created a network architecture capable of learning , backprop implemented for sigmoid and relu 
# forward and backward pass enabled , minimal working network architecture from scratch

'''problems:

   Biases and Batches not compatible
   Initialization of weights NOT optimal
   hardcoded components that should rather be dynamic
   softmax incapable of learning AND no cross enropy loss implemented here

   Non Linear and Linear can both be Integrated into one class , we kept it sepeate Till now for Readability and modularity.

'''


MIN_VAL = 0
MAX_VAL = 1


class LinearOutput:
    def __init__(self , num_inputs : int, num_outputs : int , activation : str):

        self.weight_matrix = np.random.uniform(MIN_VAL , MAX_VAL , size=(num_inputs,num_outputs))
        self.non_linear = NonLinear(activation)

    
    def input_check(self , inputs : np.typing.NDArray):
        
        if self.weight_matrix.shape[0] != inputs.shape[1]:
            raise Exception("Invalid size")

    def forward(self, inputs : np.typing.NDArray):
        self.input_check(inputs)
              
        self.inputs = inputs
        linear_outputs = np.dot(self.inputs , self.weight_matrix)
        self.non_linear_outputs = self.non_linear.forward(linear_outputs)

        return self.non_linear_outputs
    
    def backprop(self , non_linear_gradients : np.typing.NDArray , learning_rate):

        # Chnaging weights
        linear_gradients = self.non_linear.backprop(non_linear_gradients)

        # transpose of inputs needed for backprop
        weight_changes = np.dot(self.inputs.T,linear_gradients) # dL/da*da/dz*inputs - inputs = dz/dw
        # Modifying weights -> learning 
        self.weight_matrix -= weight_changes*learning_rate

        # Calculating non linear gradient for previous layer , transpose for backprop
        prev_layer_grads =  np.dot(linear_gradients,self.weight_matrix.T)
        # returning the non linear grads for backprop for previous layer
        return prev_layer_grads 



    


class NonLinear:
    def __init__(self , activation_function: str):
        self.activation = activation_function

    def forward(self,linear_outputs):

        activations = ['relu' , 'sigmoid' , 'softmax']
        if self.activation.lower() not in activations:
            raise Exception("Invalid Activation Function")
        
        # inputs to non linear layer saved
        self.linear_outputs = linear_outputs

        # inputs to non linear layers used in derivative calculations
        non_linear_outputs = None
        match self.activation.lower():
            case 'sigmoid':
                non_linear_outputs = 1 / (1 + (np.exp(-linear_outputs)))
                self.sigmoid_outs = non_linear_outputs
            
            case 'relu':
                non_linear_outputs = np.maximum(0, linear_outputs)
                
            case 'softmax':

                shifted_outputs = linear_outputs - np.max(linear_outputs)
                exp_values = np.exp(shifted_outputs)

                summation = np.sum(exp_values)
                non_linear_outputs = exp_values / summation
            
        return non_linear_outputs 

    def backprop(self , non_linear_gradients , loss = None): #dL/da

        linear_transformation = 0
        match self.activation.lower():
            case 'sigmoid':
                linear_transformation = self.sigmoid_outs * (1 - self.sigmoid_outs) # da/dz
            case 'relu':
                linear_transformation = (self.linear_outputs > 0 ).astype(float) # da/dz
            case 'softmax':
                pass # for now , gradient unclear

        return non_linear_gradients*linear_transformation  # dL/da*da/dz
        
        
        


class Network:
    def __init__(self , loss_function):
        self.temp_layers = []
        self.loss_fn = loss_function
    
    def Linear(self , num_inputs , num_outputs , activation):
        self.temp_layers.append(LinearOutput(num_inputs , num_outputs , activation))
    
    def layer_connectivity_validation(self):

        if not self.temp_layers:
            raise Exception("Empty Network")

        for i in range(len(self.temp_layers) - 1):
            current_layer = self.temp_layers[i]
            next_layer = self.temp_layers[i+1]


            current_output_dim = current_layer.weight_matrix.shape[1]
            next_input_dim = next_layer.weight_matrix.shape[0]

            if current_output_dim != next_input_dim:
                raise Exception(
                    f"Shape Mismatch between Layer {i} and Layer {i+1}: "
                    f"Layer {i} outputs {current_output_dim}, "
                    f"but Layer {i+1} expects {next_input_dim}."
                )
            
        
        
    
    def loss_calc(self , true_outputs , predicted_outputs):
        valid_losses = ['bceloss' , 'mseloss']
        if self.loss_fn.lower() not in valid_losses:
            raise Exception("Invalid Loss Function")
        
        loss = None
        loss_gradient = None
        match self.loss_fn.lower():
            case 'bceloss':
                epsilon = 1e-7
               
                predicted_outputs = np.clip(predicted_outputs, epsilon, 1 - epsilon)
                
                loss = -np.mean(true_outputs*np.log(predicted_outputs) + (1-true_outputs)*np.log(1-predicted_outputs))
                
                
                N = predicted_outputs.size
                loss_gradient = ((predicted_outputs - true_outputs) / (predicted_outputs*(1 - predicted_outputs))) / N             
            
            case 'mseloss':
                loss  = np.mean((predicted_outputs - true_outputs)**2)

                N = predicted_outputs.size 
                loss_gradient = 2 * (predicted_outputs - true_outputs) / N
                
            
        return loss , loss_gradient




    def forward(self , inputs , true_outputs):
        # validating connectivity
        self.layer_connectivity_validation() 
        self.layers = np.array(self.temp_layers) 
        

        predicted_outputs = inputs 
        for layer in self.layers:
            predicted_outputs = layer.forward(predicted_outputs)

        # calculating for final model loss that is to be backpropogated
        loss , loss_gradients = self.loss_calc(true_outputs , predicted_outputs )
        
        return predicted_outputs , loss , loss_gradients
    

    
    def backward(self , loss_gradients , learning_rate):
        # initializing the backprop 
        # these loss gradients are non linear , that are first filtered to be linear for current layer and  
        # then later used  to calculate non linear errors for previous layer 
        for layer in reversed(self.layers):
            loss_gradients = layer.backprop(loss_gradients,learning_rate)


        


        

       
        


