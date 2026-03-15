import numpy as np

# Integrating Linear and Non linear functionalities into a single class 
# Adding a loss class for modularity and structure 
# we will keep the optimizer vanilla gradient discent for now

'No support for batches yet'
'Hardcoded components reduced'
'training with both weights and biases added'


class Transformation: 
    # NOT exactly a layer with respect to the traditional definition of a layer in neural networks , but more of the Area Between 2 Layers 
    # (inputs to previous layer) Inputs -> Processing (Mathematics) -> Outputs (inputs to next layer) 

    # Defining valid Activations across all instances via shared variable
    valid_activations = ('relu' , 'sigmoid' , 'softmax' , None)

    def __init__(self, input_dim, output_dim , activation = None ):
        self.activation_fn = activation
        self.initialization(in_size=input_dim , out_size=output_dim)

    def validation(self):

        if self.activation_fn.lower() not in Transformation.valid_activations:
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
                self.sigmoid_outs = Activated_Outputs   

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
    
    # === Forward pass Done for current layer ====

    def backprop(self , non_linearized_losses , learning_rate):

        linearized_losses = self.linearization(non_linearized_losses)

        # Preparing Non Linear Losses to be sent to previous layer 
        non_linearized_prev = np.dot(linearized_losses , self.weights.T) # Transpose is important

        # Training / Modifying parameters
        # linearize losses = dL/db , linearized losses * inputs = dL/dw
        # we know that dz/dw = inputs , hence we saved self.inputs
        change_in_weights = np.dot(self.inputs.T,linearized_losses) *learning_rate 
        change_in_bias = np.sum(linearized_losses) * learning_rate
        self.weights -= change_in_weights 
        self.bias -= change_in_bias

        return non_linearized_prev # dL/da for previous layer

        
    def linearization(self,non_linearized_losses): # non linearized losses - dL/da

        linearized_losses = None # da/dz 
        # since we integrated CEloss with softmax , linearization is already done in case of softmax + Cross Entropy Loss
        match self.activation_fn.lower():

            case 'sigmoid':
                # for simplicity , we just save sigmoid outputs
                linearized_losses = self.sigmoid_outs*(1 - self.sigmoid_outs)
            case 'relu':
                # we saved linear outputs of the layer for this
                linearized_losses = (self.forward_outs > 0 ).astype(float)
            case None | 'softmax': # assuming that softmax is only used with CELoss here
                linearized_losses = non_linearized_losses

        # To modify bias , dL/db = dL/da*da/dz*dz/db
        # since z = i*w + b , dz/db = 1  hence sum of linearized losses = dz/db (changes in bias)
        return non_linearized_losses*linearized_losses #dL/da*da/dz 



class Loss:

    # Adding mode - training / testing so the model does not have to calculate gradients while testing(as weights wont be modified in testing)
    def __init__(self , true_outputs , mode = 'train' ):
        
        # we can modify mode seperately , in testing mode , all we do is calculate model loss and Stop
        self.true = true_outputs
        self.mode = mode

    

    # Integrating Softmax within CELoss for not dealing with the complex jacobian matrix of softmax
    def CrossEntropyLoss(self, model_outputs):
        # to prevent log(0)
        epsilon = 1e-15
        clipped_preds = np.clip(model_outputs, epsilon, 1 - epsilon)
        
        # N is the number of samples
        N = model_outputs.shape[0] if len(model_outputs.shape) > 1 else 1

        if self.mode == 'train':
            # Sum across classes, average across the batch
            cost = -np.sum(self.true * np.log(clipped_preds)) / N
            # The simplified combined gradient for Softmax + CCE (linearized losses)
            losses = (clipped_preds - self.true) / N
            return [cost, losses]
            
        elif self.mode == 'test':
            cost = -np.sum(self.true * np.log(clipped_preds)) / N
            return [cost, None]
            
        else:
            raise Exception("Invalid mode")
        

    def BCELoss(self, model_outputs):
        epsilon = 1e-15
        clipped_preds = np.clip(model_outputs, epsilon, 1 - epsilon)
        

        N = model_outputs.size

        if self.mode == 'train':

            cost = -np.mean(self.true * np.log(clipped_preds) + (1 - self.true) * np.log(1 - clipped_preds))
            losses = ((clipped_preds - self.true) / (clipped_preds * (1 - clipped_preds))) / N
            return [cost, losses]
            
        elif self.mode == 'test':
            cost = -np.mean(self.true * np.log(clipped_preds) + (1 - self.true) * np.log(1 - clipped_preds))
            return [cost, None]
            
        else:
            raise Exception("Invalid mode")
        

    def MSELoss(self,model_outputs):
        
        if self.mode == 'train':
            cost = np.mean((model_outputs-self.true)**2)/2
            losses = (model_outputs-self.true)/model_outputs.size
            return [cost,losses]
        elif self.mode == 'test':
            cost = np.mean((model_outputs-self.true)**2)/2
            return [cost,None]
        else:
            raise Exception("Invalid mode")
        

        

class Network:
    def __init__(self):
        self.temp_layers = []
        "self.mode = 'train'" # look forward into this

    def Layer(self , num_inputs , num_outputs , activation = None):
        self.temp_layers.append(Transformation(num_inputs,num_outputs,activation))

    def network_validation(self):
        for i in range(len(self.temp_layers)-1):
            if self.temp_layers[i].weights[1] != self.temp_layers[i+1].weights[0]:
                raise Exception(f"Shape Mismatch between Layer {i} and Layer {i+1}: ")
    
    def forward(self,network_inputs):

        self.network_validation() # to check the mapping of previous outpus to next inputs
        self.layers = np.array(self.temp_layers)

        # Forward pass
        data = network_inputs
        for layer in self.layers:
            # data moves across all layers to give final model output
            data = layer.forward(data)
        
        return data
    
    def backward(self , loss_fn):
        match loss_fn.lower():
            case 'celoss': 
                pass
            case 'bceloss':
                pass
            case 'mseloss':
                pass
            case _:
                pass
        
        'cost , losses = loss[0] , loss[1]' # look forward into this 

        





        









    

