import numpy as np

MIN_VAL = 0
MAX_VAL = 1


# not implementing batches yet 
# Backpropogation Can be implemented with lesser complexity
'Matrix implementation of weights added for faster calculations , ready for forward propogation without batches'

'problem:'
'fatal missing components for backprop , no bias terms or support for batches'



class LinearOutput:
    def __init__(self , num_inputs : int, num_outputs : int , activation : str):
        # Weight matrix
        self.weight_matrix = np.random.uniform(MIN_VAL , MAX_VAL , size=(num_inputs,num_outputs))
        self.non_linear = NonLinear(activation)

    
    def input_check(self , inputs : np.typing.NDArray):
        # Here we check number of rows in a matrix (number of neurons) to number of columns in the 1D input array , which is the number of inputs , that is the number of weights for the prev layer
        if self.weight_matrix.shape[0] != inputs.shape[1]:
            raise Exception("Invalid size")

    def forward(self, inputs : np.typing.NDArray):
        self.input_check(inputs)
        
        # saved inputs , final non linear output
        self.inputs = inputs
        linear_outputs = np.dot(self.inputs , self.weight_matrix)
        self.non_linear_outputs = self.non_linear.forward(linear_outputs)

        return self.non_linear_outputs
    
    def backprop(self , incoming_linear_loss : np.typing.NDArray):
        pass # we will compute non linear loss and convert it to Lonear loss with NonLinear object
        
        # self.weight_matrix -= losses_matrix  -> matrix that holds losses for all weights in the blackbox
        # return outgoing_linear_loss



class NonLinear:
    def __init__(self , activation_function: str):
        self.activation = activation_function

    def forward(self,linear_outputs):

        activations = ['relu' , 'sigmoid' , 'softmax']
        if self.activation.lower() not in activations:
            raise Exception("Invalid Activation Function")
        
        # saved the inputs to non linear component for dL/dw = dL/da*da/dz*inputs 
        self.linear_outputs = linear_outputs

        outputs = None
        match self.activation.lower():
            case 'sigmoid':
                outputs = 1 / (1 + (np.exp(-linear_outputs)))
            
            case 'relu':
                outputs = np.maximum(0, linear_outputs)
                
            case 'softmax':

                shifted_outputs = linear_outputs - np.max(linear_outputs)
                exp_values = np.exp(shifted_outputs)

                summation = np.sum(exp_values)
                outputs = exp_values / summation
            
        return outputs

    def backprop(self , non_linear_loss , loss = None):
        pass
        # return linear_loss


class Network:
    def __init__(self , loss_function):
        self.temp_layers = []
        self.loss_fn = loss_function
    
    def Linear(self , num_inputs , num_outputs , activation):
        self.temp_layers.append(LinearOutput(num_inputs , num_outputs , activation))
    
    def layer_connectivity_validation(self):
        """
        Validates that the output dimension of each layer matches 
        the input dimension of the subsequent layer.
        """
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
                # Clamp values to prevent crash
                predicted_outputs = np.clip(predicted_outputs, epsilon, 1 - epsilon)
                
                loss = -np.mean(true_outputs*np.log(predicted_outputs) + (1-true_outputs)*np.log(1-predicted_outputs))
                
                # Don't forget to divide by N here too, since loss uses np.mean!
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

        loss , loss_gradients = self.loss_calc(true_outputs , predicted_outputs )
        
        return predicted_outputs , loss , loss_gradients
    
    def backward(self):
        pass

        



        

        