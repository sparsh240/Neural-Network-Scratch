import numpy as np

# Integrating Linear and Non linear functionalities into a single class 
# Adding a loss class for modularity and structure 
# we will keep the optimizer vanilla gradient discent for now
'Hardcoded components reduced'
'training with both weights and biases added'


'No batches supported'
class Transformation: 
    # NOT exactly a layer with respect to the traditional definition of a layer in neural networks , but more of the Area Between 2 Layers 
    # (inputs to previous layer) Inputs -> Processing (Mathematics) -> Outputs (inputs to next layer) 

    # Defining valid Activations across all instances via shared variable
    valid_activations = ('relu' , 'sigmoid' , 'softmax' , 'none')

    def __init__(self, input_dim, output_dim , activation = None ):
        self.activation_fn = activation
        self.initialization(in_size=input_dim , out_size=output_dim)

    def validation(self):

        if str(self.activation_fn).lower() not in Transformation.valid_activations:
            raise Exception("Not a Valid Activation Function")
        if self.inputs.shape[0] != self.weights.shape[0]:
            raise Exception("Data Shape Not same as expected Input shape")
        
    # Uniform
    def initialization(self, in_size , out_size): # initializing weight and biases 
        # Conditionally define weight matrix based on input-output sizes and weight matrix
        self.weights = 0
        self.bias = 0
        match str(self.activation_fn).lower():
            case 'sigmoid': # xavier initialization 
                self.weights = np.random.uniform(-np.sqrt(6/(in_size+out_size)),np.sqrt(6/(in_size+out_size)) , size=(in_size,out_size))
                self.bias = np.zeros(out_size)
            case 'relu': # He initialization
                self.weights = np.random.uniform(-np.sqrt(6/(in_size)),np.sqrt(6/(in_size)) , size=(in_size,out_size))
                self.bias = np.full(out_size,0.01) # to prevent neurons from being dead at the start of the training  
            case 'softmax': # xavier (usually suggested)
                self.weights = np.random.uniform(-np.sqrt(6/(in_size+out_size)),np.sqrt(6/(in_size+out_size)) , size=(in_size,out_size))
                self.bias = np.zeros(out_size)

            case 'none':
                self.weights = np.random.uniform(-1 , 1, size=(in_size,out_size))
                self.bias = np.zeros(out_size)


    def forward(self, inputs:np.typing.NDArray):
        self.inputs = inputs
        self.validation()
        # Performs forward pass through layer
        linear_outs = np.dot(self.inputs, self.weights) + self.bias # Linear pass 
        self.forward_outs = self.activation(linear = linear_outs) # Activation
        return self.forward_outs # Forwarded to next layer as inputs
    
    def activation(self , linear):

        Activated_Outputs = None
        match str(self.activation_fn).lower():
            
            case 'sigmoid':
                Activated_Outputs = 1 / (1 + (np.exp(-linear))) 
                self.sigmoid_outs = Activated_Outputs   

            case 'relu':
                Activated_Outputs = np.maximum(0,linear)

            case 'softmax':
                # For NOT Computationally destroying anything
                shift = linear - np.max(linear) 
                exponents = np.exp(shift)
                summation = np.sum(exponents)
                Activated_Outputs = exponents/summation
            
            case 'none':
                Activated_Outputs = linear

        if Activated_Outputs is None:
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
        change_in_weights = np.outer(self.inputs,linearized_losses) *learning_rate 
        change_in_bias = linearized_losses * learning_rate # axis = 0 is important
        self.weights -= change_in_weights 
        self.bias -= change_in_bias

        return non_linearized_prev # dL/da for previous layer

        
    def linearization(self,non_linearized_losses): # non linearized losses - dL/da

        linearized_losses = None # da/dz 
        # since we integrated CEloss with softmax , linearization is already done in case of softmax + Cross Entropy Loss
        match str(self.activation_fn).lower():

            case 'sigmoid':
                # for simplicity , we just save sigmoid outputs
                linearized_losses = self.sigmoid_outs*(1 - self.sigmoid_outs)
            case 'relu':
                # we saved linear outputs of the layer for this
                linearized_losses = (self.forward_outs > 0 ).astype(float)
            case 'none' | 'softmax': # assuming that softmax is only used with CELoss here
                linearized_losses = 1 # since we return non linearized losses * linearized losses , 1 keeps it as is.

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
    def __init__(self,loss_fn):
        self.temp_layers = []
        self.mode = 'train'
        self.loss_fn = loss_fn
        self.validation_check = 0

    def Layer(self , num_inputs , num_outputs , activation = None):
        self.temp_layers.append(Transformation(num_inputs,num_outputs,activation))

    def network_validation(self):
        for i in range(len(self.temp_layers)-1):
            if self.temp_layers[i].weights.shape[1] != self.temp_layers[i+1].weights.shape[0]:
                raise Exception(f"Shape Mismatch between Layer {i} and Layer {i+1}: ")
    
    def forward(self,network_inputs):
        # to make sure not running valiadation on each iteration
        if self.validation_check == 0:
            self.network_validation() # to check the mapping of previous outpus to next inputs
            self.validation_check = 1
        self.layers = np.array(self.temp_layers)

        # Forward pass
        data = network_inputs
        for layer in self.layers:
            # data moves across all layers to give final model output
            data = layer.forward(data)
        
        return data
    
    def backward(self ,true_outputs,model_outputs,learning_rate):
        loss = Loss(true_outputs ,self.mode) # mode can be modified externally
        match self.loss_fn.lower():
           
            case 'celoss':  # initial losses
                initial = loss.CrossEntropyLoss(model_outputs)
            case 'bceloss':
                initial = loss.BCELoss(model_outputs)
            case 'mseloss':
                initial = loss.MSELoss(model_outputs)
            case _:
                raise Exception("Invalid Loss Function")
        
        cost , losses = initial[0] , initial[1] 

        # Backpropogating to all layers
        for layer in reversed(self.layers):
            losses = layer.backprop(losses , learning_rate) # losses of prev layer calculated with current layer's losses

        return cost
         
