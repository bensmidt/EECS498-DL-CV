# PyTorch: torch.nn

*Disclaimer: much of this document is pulled directly from [CS231N: Assignment 2 PyTorch](http://cs231n.stanford.edu/schedule.html). I'm synthesizing it for my own reference and to help me remember.*

This document focuses on using `torch.nn` to build neural networks. `torch.nn` has an abundant set of modules which we can think of as different layers. There are three different layers of abstraction we'll be using in PyTorch: Barebones PyTorch, `nn.module` API, and the `nn.sequential` API. 

- Barebones: 

        highly flexible - quite inconvenient
- nn.Module API: 

        highly flexible - somewhat convenient
- nn.Sequential API: 

        inflexible - highly convenient

## Table of Contents
1. [Barebones Pytorch](#barebones-pytorch)
     - [Gradients](#gradients)
     - [Auto-Softmax](#auto-softmax)
     - [Three Layer ConvNet](#three-layer-convnet)
1. [PyTorch Module API](#pytorch-module-api)
1. [PyTorch Sequential API](#pytorch-sequential-api)
-  [ResNet Implementation](#resnet)
-  [NN.Functional](#nnfunctional)

## Barebones PyTorch
### Gradients
If a given tensor needs to be backpropagated, we create that tensor with `tensor.requires_grad == True`, PyTorch then stores the gradient in an attribute of the tensor `tensor.grad`

### Auto-Softmax
You do not need to add your softmax activation after your last fully connected layer. PyTorch's cross entropy loss automatically does it for you. This bundling step actually makes the computation more efficient.

### Three Layer ConvNet
The following are implementations of initializing your parameters for a ConvNet as well as creating the network. I know above we said that this is inconvenient, but compared to what we've been doing in [EECS498](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2019/schedule.html), this is amazinggg. 

[NN.Conv2d](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.conv2d)

Notice when checking accuracy we use `with: torch.no_grad()` so that PyTorch doesn't build a computational graph with the tensors used for this function.

<details close>
<summary>Define and Initialize the Network</summary>

```python
def initialize_three_layer_conv_part2(dtype=torch.float, device='cpu'):
    '''
    Initializes weights for the three_layer_convnet for part II
    Inputs:
    - dtype: A torch data type object; all computations will be performed using
        this datatype. float is faster but less accurate, so you should use
        double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
    '''
    # Input/Output dimenssions
    C, H, W = 3, 32, 32
    num_classes = 10

    # Hidden layer channel and kernel sizes
    channel_1 = 32
    channel_2 = 16
    kernel_size_1 = 5
    kernel_size_2 = 3

    # Initialize the weights
    conv_w1 = None
    conv_b1 = None
    conv_w2 = None
    conv_b2 = None
    fc_w = None
    fc_b = None

    ##############################################################################
    # TODO: Define and initialize the parameters of a three-layer ConvNet           
    # using nn.init.kaiming_normal_. You should initialize your bias vectors    
    # using the zero_weight function.                         
    # You are given all the necessary variables above for initializing weights. 
    ##############################################################################
    conv_w1 = nn.init.kaiming_normal_( torch.empty(channel_1, C, kernel_size_1, kernel_size_1, dtype=dtype, device=device) )
    conv_w1.requires_grad=True
    conv_b1 = nn.init.zeros_( torch.empty(channel_1, dtype=dtype, device=device) )
    conv_b1.requires_grad=True
    conv_w2 = nn.init.kaiming_normal_( torch.empty(channel_2, channel_1, kernel_size_2, kernel_size_2, dtype=dtype, device=device) )
    conv_w2.requires_grad=True
    conv_b2 = nn.init.zeros_( torch.empty(channel_2, dtype=dtype, device=device) )
    conv_b2.requires_grad=True
    fc_w = nn.init.kaiming_normal_( torch.empty(num_classes, H*W*channel_2, dtype=dtype, device=device) )
    fc_w.requires_grad=True
    fc_b = nn.init.zeros_( torch.empty(num_classes, dtype=dtype, device=device) )
    fc_b.requires_grad=True
    ##############################################################################
    #                                 END OF YOUR CODE                            
    ##############################################################################
    return [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
```
</details>




<details close>
<summary>Build the Network</summary>

```python
def three_layer_convnet(x, params):
    """
    Performs the forward pass of a three-layer convolutional network with the
    architecture defined above.

    Inputs:
    - x: A PyTorch Tensor of shape (N, C, H, W) giving a minibatch of images
    - params: A list of PyTorch Tensors giving the weights and biases for the
        network; should contain the following:
        - conv_w1: PyTorch Tensor of shape (channel_1, C, KH1, KW1) giving weights
        for the first convolutional layer
        - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
        convolutional layer
        - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
        weights for the second convolutional layer
        - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
        convolutional layer
        - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
        figure out what the shape should be?
        - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
        figure out what the shape should be?

    Returns:
    - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
    """
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None
    ##############################################################################
    # TODO: Implement the forward pass for the three-layer ConvNet.              
    # The network have the following architecture:                               
    # 1. Conv layer (with bias) with 32 5x5 filters, with zero-padding of 2     
    #   2. ReLU                                                                  
    # 3. Conv layer (with bias) with 16 3x3 filters, with zero-padding of 1     
    # 4. ReLU                                                                   
    # 5. Fully-connected layer (with bias) to compute scores for 10 classes    
    # Hint: F.linear, F.conv2d, F.relu, flatten (implemented above)                                   
    ##############################################################################
    h1 = F.conv2d(input=x, weight=conv_w1, bias=conv_b1, padding=2)
    h1 = F.relu(h1)
    h2 = F.conv2d(input=h1, weight=conv_w2, bias=conv_b2, padding=1)
    h2 = F.relu(h2)
    h2_flat = flatten(h2)
    scores = F.linear(input=h2_flat, weight=fc_w, bias=fc_b)
    ##############################################################################
    #                                 END OF YOUR CODE                             
    ##############################################################################
    return scores
```
</details>




<details closed>
<summary>Kaiming Initialization</summary>

```python
nn.init.kaiming_normal_(torch.empty(3, 5, dtype=to_float, device='cuda'))

nn.init.zeros_(torch.empty(3 ,5, dtype=to_float, device='cuda'))
```
</details>




<details closed>
<summary>Checking Accuracy</summary>

```python
def check_accuracy_part2(loader, model_fn, params):
    """
    Check the accuracy of a classification model.

    Inputs:
    - loader: A DataLoader for the data split we want to check
    - model_fn: A function that performs the forward pass of the model,
    with the signature scores = model_fn(x, params)
    - params: List of PyTorch Tensors giving parameters of the model

    Returns: Nothing, but prints the accuracy of the model
    """
    split = 'val' if loader.dataset.train else 'test'
    print('Checking accuracy on the %s set' % split)
    num_correct, num_samples = 0, 0
    with torch.no_grad():
    for x, y in loader:
        x = x.to(device='cuda', dtype=to_float)  # move to device, e.g. GPU
        y = y.to(device='cuda', dtype=to_long)
        scores = model_fn(x, params)
        _, preds = scores.max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
    return acc
```
</details>



<details closed>
<summary>Training Loop</summary>

```python
def train_part2(model_fn, params, learning_rate):
    """
    Train a model on CIFAR-10.

    Inputs:
    - model_fn: A Python function that performs the forward pass of the model.
    It should have the signature scores = model_fn(x, params) where x is a
    PyTorch Tensor of image data, params is a list of PyTorch Tensors giving
    model weights, and scores is a PyTorch Tensor of shape (N, C) giving
    scores for the elements in x.
    - params: List of PyTorch Tensors giving weights for the model
    - learning_rate: Python scalar giving the learning rate to use for SGD

    Returns: Nothing
    """
    for t, (x, y) in enumerate(loader_train):
    # Move the data to the proper device (GPU or CPU)
    x = x.to(device='cuda', dtype=to_float)
    y = y.to(device='cuda', dtype=to_long)

    # Forward pass: compute scores and loss
    scores = model_fn(x, params)
    loss = F.cross_entropy(scores, y)

    # Backward pass: PyTorch figures out which Tensors in the computational
    # graph has requires_grad=True and uses backpropagation to compute the
    # gradient of the loss with respect to these Tensors, and stores the
    # gradients in the .grad attribute of each Tensor.
    loss.backward()

    # Update parameters. We don't want to backpropagate through the
    # parameter updates, so we scope the updates under a torch.no_grad()
    # context manager to prevent a computational graph from being built.
    with torch.no_grad():
        for w in params:
        if w.requires_grad:
            w -= learning_rate * w.grad

            # Manually zero the gradients after running the backward pass
            w.grad.zero_()

    if t % 100 == 0 or t == len(loader_train)-1:
        print('Iteration %d, loss = %.4f' % (t, loss.item()))
        acc = check_accuracy_part2(loader_val, model_fn, params)
    return acc
```
</details>

<details closed>
<summary> Train the Network </summary>

```python

reset_seed(0)
learning_rate = 3e-3
# YOUR_TURN: Impelement the initialize_three_layer_conv_part2 function
params = initialize_three_layer_conv_part2(dtype=to_float, device='cuda')
acc_hist_part2 = train_part2(three_layer_convnet, params, learning_rate)
```
</details>

Our training loop uses [nn.functional.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)


## PyTorch Module API
You can see, when using barebones PyTorch we must keep track of all the tensors by hand. This, of course, is quite error prone. We use PyTorch's Module API to make life significantly easier while still retaining flexibility. 

PyTorch provides the `nn.Module` API and `torch.optim` package which implements different optimization methods such as Nesterov Momentum or Adam for us. 

Use the `nn.Module` API in three steps (straight from CS231N): 

1. Define your network as a class and subclass `nn.Module`.

2. Define all the layers you need as class attributes in `__init__()` constructor. Layer objects like `nn.Linear` and `nn.Conv2d` are themselves `nn.Module` subclasses and contain learnable parameters, so that you don't have to instantiate the raw tensors yourself. `nn.Module` tracks these internal parameters for you. Refer to the [doc](http://pytorch.org/docs/master/nn.html) to learn more about the dozens of builtin layers. 

    **Warning: don't forget to call the `super().__init__()` first!**

3. In the `forward()` method, define the *connectivity* of your network. You should use the attributes defined in `__init__` as function calls that take tensor as input and output the "transformed" tensor. Do *not* create any new layers with learnable parameters in `forward()`! All of them must be declared upfront in `__init__`. 

Note that the Check Accuracy and the Training Loop are different from the previous implementation. 

<details closed>
<summary> Three Layer ConvNet </summary>

```python
class ThreeLayerConvNet(nn.Module):

def __init__(self, in_channel, channel_1, channel_2, num_classes, device='cpu', dtype=torch.float):
    super().__init__()
    ############################################################################
    # TODO: Set up the layers you need for a three-layer ConvNet with the       
    # architecture defined below. You should initialize the weight  of the
    # model using Kaiming normal initialization, and zero out the bias vectors.     
    #                                       
    # The network architecture should be the same as in Part II:          
    #   1. Convolutional layer with channel_1 5x5 filters with zero-padding of 2  
    #   2. ReLU                                   
    #   3. Convolutional layer with channel_2 3x3 filters with zero-padding of 1
    #   4. ReLU                                   
    #   5. Fully-connected layer to num_classes classes               
    #                                       
    # We assume that the size of the input of this network is `H = W = 32`, and   
    # there is no pooling; this information is required when computing the number  
    # of input channels in the last fully-connected layer.              
    #                                         
    # HINT: nn.Conv2d, nn.init.kaiming_normal_, nn.init.zeros_            
    ############################################################################
    self.conv1 = nn.Conv2d(in_channels=in_channel, 
                            out_channels=channel_1, 
                            kernel_size=5, 
                            padding=2, 
                            device=device, dtype=dtype)
    self.conv2 = nn.Conv2d(in_channels=channel_1, 
                            out_channels=channel_2, 
                            kernel_size=3, 
                            padding=1, 
                            bias=True, 
                            device=device, dtype=dtype)
    self.fc3 = nn.Linear(in_features=channel_2*32*32, 
                            out_features=num_classes, 
                            device=device, dtype=dtype)
    ############################################################################
    #                           END OF YOUR CODE                            
    ############################################################################

def forward(self, x):
    scores = None
    ############################################################################
    # TODO: Implement the forward function for a 3-layer ConvNet. you      
    # should use the layers you defined in __init__ and specify the       
    # connectivity of those layers in forward()   
    # Hint: flatten (implemented at the start of part II)                          
    ############################################################################
    h1 = F.relu(self.conv1(x))
    h2 = F.relu(self.conv2(h1))
    scores = self.fc3(flatten(h2))
    ############################################################################
    #                            END OF YOUR CODE                          
    ############################################################################
    return scores
```
</details>

<details closed>
<summary> Initializating ConvNet</summary>

```python

def initialize_three_layer_conv_part3():
    '''
    Instantiates a ThreeLayerConvNet model and a corresponding optimizer for part III
    '''

    # Parameters for ThreeLayerConvNet
    C = 3
    num_classes = 10

    channel_1 = 32
    channel_2 = 16

    # Parameters for optimizer
    learning_rate = 3e-3
    weight_decay = 1e-4

    model = None
    optimizer = None
    ##############################################################################
    # TODO: Instantiate ThreeLayerConvNet model and a corresponding optimizer.     
    # Use the above mentioned variables for setting the parameters.                
    # You should train the model using stochastic gradient descent without       
    # momentum, with L2 weight decay of 1e-4.                    
    ##############################################################################
    model = ThreeLayerConvNet(C, channel_1, channel_2, num_classes, device='cuda')

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    ##############################################################################
    #                                 END OF YOUR CODE                            
    ##############################################################################
    return model, optimizer
```
</details>

<details closed>
<summary> Check Accuracy </summary>

```python

def check_accuracy_part34(loader, model):
  if loader.dataset.train:
    print('Checking accuracy on validation set')
  else:
    print('Checking accuracy on test set')   
  num_correct = 0
  num_samples = 0
  model.eval()  # set model to evaluation mode
  with torch.no_grad():
    for x, y in loader:
      x = x.to(device='cuda', dtype=to_float)  # move to device, e.g. GPU
      y = y.to(device='cuda', dtype=to_long)
      scores = model(x)
      _, preds = scores.max(1)
      num_correct += (preds == y).sum()
      num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
  return acc
```

</details>

<details closed>
<summary> Training Loop </summary>

```python

def adjust_learning_rate(optimizer, lrd, epoch, schedule):
    """
    Multiply lrd to the learning rate if epoch is in schedule

    Inputs:
    - optimizer: An Optimizer object we will use to train the model
    - lrd: learning rate decay; a factor multiplied at scheduled epochs
    - epochs: the current epoch number
    - schedule: the list of epochs that requires learning rate update

    Returns: Nothing, but learning rate might be updated
    """
    if epoch in schedule:
    for param_group in optimizer.param_groups:
        print('lr decay from {} to {}'.format(param_group['lr'], param_group['lr'] * lrd))
        param_group['lr'] *= lrd

def train_part345(model, optimizer, epochs=1, learning_rate_decay=.1, schedule=[], verbose=True):
  """
  Train a model on CIFAR-10 using the PyTorch Module API.
  
  Inputs:
  - model: A PyTorch Module giving the model to train.
  - optimizer: An Optimizer object we will use to train the model
  - epochs: (Optional) A Python integer giving the number of epochs to train for
  
  Returns: Nothing, but prints model accuracies during training.
  """
  model = model.to(device='cuda')  # move the model parameters to CPU/GPU
  num_iters = epochs * len(loader_train)
  print_every = 100
  if verbose:
    num_prints = num_iters // print_every + 1
  else:
    num_prints = epochs
  acc_history = torch.zeros(num_prints, dtype=to_float)
  iter_history = torch.zeros(num_prints, dtype=to_long)
  for e in range(epochs):
    
    adjust_learning_rate(optimizer, learning_rate_decay, e, schedule)
    
    for t, (x, y) in enumerate(loader_train):
      model.train()  # put model to training mode
      x = x.to(device='cuda', dtype=to_float)  # move to device, e.g. GPU
      y = y.to(device='cuda', dtype=to_long)

      scores = model(x)
      loss = F.cross_entropy(scores, y)

      # Zero out all of the gradients for the variables which the optimizer
      # will update.
      optimizer.zero_grad()

      # This is the backwards pass: compute the gradient of the loss with
      # respect to each  parameter of the model.
      loss.backward()

      # Actually update the parameters of the model using the gradients
      # computed by the backwards pass.
      optimizer.step()

      tt = t + e * len(loader_train)

      if verbose and (tt % print_every == 0 or (e == epochs-1 and t == len(loader_train)-1)):
        print('Epoch %d, Iteration %d, loss = %.4f' % (e, tt, loss.item()))
        acc = check_accuracy_part34(loader_val, model)
        acc_history[tt // print_every] = acc
        iter_history[tt // print_every] = tt
        print()
      elif not verbose and (t == len(loader_train)-1):
        print('Epoch %d, Iteration %d, loss = %.4f' % (e, tt, loss.item()))
        acc = check_accuracy_part34(loader_val, model)
        acc_history[e] = acc
        iter_history[e] = tt
        print()
  return acc_history, iter_history
```
</details>




<details closed>
<summary> Train the Network </summary>

```python 
reset_seed(0)
# YOUR_TURN: Impelement initialize_three_layer_conv_part3
model, optimizer = initialize_three_layer_conv_part3()
acc_hist_part3, _ = train_part345(model, optimizer)
```
</details>




## PyTorch Sequential API
If your model is fairly simple, PyTorch provides an extremely convenient API with `nn.Sequential` that allows you to quickly define a network. 

<details closed>
<summary> Three Layer ConvNet Class </summary>

```python
class Flatten(nn.Module):
  def forward(self, x):
    return flatten(x)


def initialize_three_layer_conv_part4():
  '''
  Instantiates a ThreeLayerConvNet model and a corresponding optimizer for part IV
  '''
  # Input/Output dimenssions
  C, H, W = 3, 32, 32
  num_classes = 10

  # Hidden layer channel and kernel sizes
  channel_1 = 32
  channel_2 = 16
  kernel_size_1 = 5
  pad_size_1 = 2
  kernel_size_2 = 3
  pad_size_2 = 1

  # Parameters for optimizer
  learning_rate = 1e-2
  weight_decay = 1e-4
  momentum = 0.5

  model = None
  optimizer = None
  ##################################################################################
  # TODO: Rewrite the 3-layer ConvNet with bias from Part III with Sequential API and 
  # a corresponding optimizer.
  # You don't have to re-initialize your weight matrices and bias vectors.  
  # Here you should use `nn.Sequential` to define a three-layer ConvNet with:
  #   1. Convolutional layer (with bias) with 32 5x5 filters, with zero-padding of 2 
  #   2. ReLU                                      
  #   3. Convolutional layer (with bias) with 16 3x3 filters, with zero-padding of 1 
  #   4. ReLU                                      
  #   5. Fully-connected layer (with bias) to compute scores for 10 classes        
  #                                            
  # You should optimize your model using stochastic gradient descent with Nesterov   
  # momentum 0.5, with L2 weight decay of 1e-4 as given in the variables above.   
  # Hint: nn.Sequential, Flatten (implemented at the start of Part IV)   
  ####################################################################################
  model  = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(C, channel_1, kernel_size=5, padding=2)), 
    ('relu1', nn.ReLU()), 
    ('conv2', nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1)), 
    ('relu2', nn.ReLU()), 
    ('flatten', Flatten()), 
    ('fc3', nn.Linear(channel_2*H*W, num_classes)), 
  ]))

  optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                        weight_decay=weight_decay, 
                        momentum=momentum, nesterov=True)
  ################################################################################
  #                                 END OF YOUR CODE                             
  ################################################################################
  return model, optimizer
```
</details>



<details closed>
<summary> Train the Network </summary>

```python
reset_seed(0)

# YOUR_TURN: Impelement initialize_three_layer_conv_part4
model, optimizer = initialize_three_layer_conv_part4()
print('Architecture:')
print(model) # printing `nn.Module` shows the architecture of the module.

acc_hist_part4, _ = train_part345(model, optimizer)
```
</details>

## ResNet

## NN.Functional
 - Spatial Batch Normalization: [BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d)
 - ConvNet: [NN.Conv2d](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.conv2d)
 - 