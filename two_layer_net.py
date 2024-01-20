"""
Implements a two-layer Neural Network classifier in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
import statistics
from typing import Dict, List, Callable, Optional


def hello_two_layer_net():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from two_layer_net.py!")


# Template class modules that we will use later: Do not edit/modify this class
class TwoLayerNet(object):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        std: float = 1e-4,
    ):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        - dtype: Optional, data type of each initial weight params
        - device: Optional, whether the weight params is on GPU or CPU
        - std: Optional, initial weight scaler.
        """
        # reset seed before start
        random.seed(0)
        torch.manual_seed(0)

        self.params = {}
        self.params["W1"] = std * torch.randn(
            input_size, hidden_size, dtype=dtype, device=device
        )
        self.params["b1"] = torch.zeros(hidden_size, dtype=dtype, device=device)
        self.params["W2"] = std * torch.randn(
            hidden_size, output_size, dtype=dtype, device=device
        )
        self.params["b2"] = torch.zeros(output_size, dtype=dtype, device=device)

    def loss(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        reg: float = 0.0,
    ):
        return nn_forward_backward(self.params, X, y, reg)

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        learning_rate: float = 1e-3,
        learning_rate_decay: float = 0.95,
        reg: float = 5e-6,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        # fmt: off
        return nn_train(
            self.params, nn_forward_backward, nn_predict, X, y,
            X_val, y_val, learning_rate, learning_rate_decay,
            reg, num_iters, batch_size, verbose,
        )
        # fmt: on

    def predict(self, X: torch.Tensor):
        return nn_predict(self.params, nn_forward_backward, X)

    def save(self, path: str):
        torch.save(self.params, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")
        self.params = checkpoint
        if len(self.params) != 4:
            raise Exception("Failed to load your checkpoint")

        for param in ["W1", "b1", "W2", "b2"]:
            if param not in self.params:
                raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


def nn_forward_pass(params: Dict[str, torch.Tensor], X: torch.Tensor):
    """
    The first stage of our neural network implementation: Run the forward pass
    of the network to compute the hidden layer features and classification
    scores. The network architecture should be:

    FC layer -> ReLU (hidden) -> FC layer (scores)

    As a practice, we will NOT allow to use torch.relu and torch.nn ops
    just for this time (you can use it from A3).

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - X: Input data of shape (N, D). Each X[i] is a training sample.

    Returns a tuple of:
    - scores: Tensor of shape (N, C) giving the classification scores for X
    - hidden: Tensor of shape (N, H) giving the hidden layer representation
      for each input value (after the ReLU).
    """
    # Unpack variables from the params dictionary
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    N, D = X.shape

    # Compute the forward pass
    hidden = None
    scores = None
    ############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input.#
    # Store the result in the scores variable, which should be an tensor of    #
    # shape (N, C).                                                            #
    ############################################################################  
    #FC1 
    s1 = torch.mm(X, W1)
    s2 = s1 + b1
    #ReLu
    s3 = torch.max(torch.tensor(0.0), s2)
    hidden = s3
    #FC2 
    s4 = torch.mm(s3, W2)
    s5 = s4 + b2 #scores output 
    scores = s5
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return scores, hidden


def nn_forward_backward(
    params: Dict[str, torch.Tensor],
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    reg: float = 0.0
):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network. When you implement loss and gradient, please don't forget to
    scale the losses/gradients by the batch size.

    Inputs: First two parameters (params, X) are same as nn_forward_pass
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a tensor scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    N, D = X.shape

    scores, h1 = nn_forward_pass(params, X)
    # If the targets are not given then jump out, we're done
    if y is None:
        return scores

    # Compute the loss
    loss = None
    ############################################################################
    # TODO: Compute the loss, based on the results from nn_forward_pass.       #
    # This should include both the data loss and L2 regularization for W1 and  #
    # W2. Store the result in the variable loss, which should be a scalar. Use #
    # the Softmax classifier loss. When you implment the regularization over W,#
    # please DO NOT multiply the regularization term by 1/2 (no coefficient).  #
    # If you are not careful here, it is easy to run into numeric instability  #
    # (Check Numeric Stability in http://cs231n.github.io/linear-classify/).   #
    ############################################################################
    # Replace "pass" statement with your code
    
    ''' Alternative code to compute backpropogation using pytorch methods
    W1 = W1.clone()
    b1 = b1.clone()
    W2 = W2.clone()
    b2 = b2.clone()
    
    W1.requires_grad_(True)
    b1.requires_grad_(True)
    W2.requires_grad_(True)
    b2.requires_grad_(True)
    
    def print_grad(grad):
        print(grad)
    '''

    #FC1 
    s1 = torch.mm(X, W1)
    s2 = s1 + b1
    #ReLu
    s3 = torch.max(torch.tensor(0.0), s2)
    #FC2 
    s4 = torch.mm(s3, W2)
    s5 = s4 + b2 #scores output 
    #Softmax
    s6 = torch.exp(s5) 
    s7 = s6.sum(dim=1, keepdim=True)
    s8 = s6/s7
    true_onehot = torch.zeros_like(scores)
    for pred in range(scores.size(0)):
        true_onehot[pred, y[pred]] = 1
    s15 = true_onehot 
    s16 = s8*s15  #softmax output
    #L2 Regularization 
    s9 = W1**2
    s10 = s9.sum()
    s11 = W2**2
    s12 = s11.sum()
    s13 = s10+s12
    s14 = reg*s13 #l2 regularization 
    #Cross entropy loss
    s17 = torch.sum(s16)
    s18 = s17/s8.size(0)
    s19 = -torch.log(s18) #cross entropy loss 
    #Total Loss 
    s20 = s19+s14
    loss = s20
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    # Backward pass: compute gradients
    grads = {}
    ###########################################################################
    # TODO: Compute the backward pass, computing the derivatives of the       #
    # weights and biases. Store the results in the grads dictionary.          #
    # For example, grads['W1'] should store the gradient on W1, and be a      #
    # tensor of same size                                                     #
    ###########################################################################
    
    
    #Total Loss
    grad_L = 1.0 
    grad_s20 = grad_L
    #Cross Entropy Loss
    grad_s19 = grad_s20
    grad_s18 = grad_s19*(-1.0/s18)
    grad_s17 = grad_s18*(1.0/s8.size(0))
    grad_s16 = grad_s17*torch.ones_like(s16)
    #not calculating grad_s15 because not necessary
    #L2 Regularization
    grad_s14 = grad_s20
    grad_s13 = grad_s14*reg 
    grad_s12 = grad_s13
    grad_s11 = grad_s12*torch.ones_like(s11)
    grad_W2 = grad_s11*2*W2
    grad_s10 = grad_s13 
    grad_s9 = grad_s10*torch.ones_like(s9)
    grad_W1 = grad_s9*2*W1
    #Softmax
    grad_s8 = grad_s16*s15
    grad_s7 = (grad_s8*(-s6/s7**2)).sum(1, keepdims=True)
    grad_s6 = grad_s7*torch.ones_like(s6) + grad_s8/s7
    #FC2
    grad_s5 = grad_s6*torch.exp(s5)
    grad_b2 = torch.sum(grad_s5, dim=0) 
    grad_s4 = grad_s5
    grad_W2 += torch.mm(s3.t(), grad_s4) 
    #ReLU
    grad_s3 = torch.mm(grad_s4, W2.t())
    #FC1
    s2_masked = s3
    s2_masked[s2_masked!=0] = 1
    grad_s2 = grad_s3*s2_masked
    grad_s1 = grad_s2
    grad_b1 = torch.sum(grad_s2, dim=0) 
    #not calculating grad_X because not necessary 
    grad_W1 += torch.mm(X.t(), grad_s1) 
    
    '''Alternative code to compute backpropogation using pytorch methods
    grad_W1 = W1.grad
    grad_b1 = b1.grad
    grad_W2 = W2.grad
    grad_b2 = b2.grad
    '''

    grads['W1'] = grad_W1
    grads['b1'] = grad_b1
    grads['W2'] = grad_W2
    grads['b2'] = grad_b2
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return loss, grads

def sample_batch(X: torch.Tensor, y: torch.Tensor, num_train: int, batch_size: int):
    """
    Sample batch_size elements from the training data and their
    corresponding labels to use in this round of gradient descent.
    """
    X_batch = None
    y_batch = None
    #########################################################################
    # TODO: Store the data in X_batch and their corresponding labels in     #
    # y_batch; after sampling, X_batch should have shape (batch_size, dim)  #
    # and y_batch should have shape (batch_size,)                           #
    #                                                                       #
    # Hint: Use torch.randint to generate indices.                          #
    #########################################################################
    # Replace "pass" statement with your code
    sample_indices = torch.randint(high=num_train, size=(batch_size,))
    X_batch = X[sample_indices]
    y_batch = y[sample_indices]
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return X_batch, y_batch

def nn_train(
    params: Dict[str, torch.Tensor],
    loss_func: Callable,
    pred_func: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    learning_rate: float = 1e-3,
    learning_rate_decay: float = 0.95,
    reg: float = 5e-6,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - loss_func: a loss function that computes the loss and the gradients.
      It takes as input:
      - params: Same as input to nn_train
      - X_batch: A minibatch of inputs of shape (B, D)
      - y_batch: Ground-truth labels for X_batch
      - reg: Same as input to nn_train
      And it returns a tuple of:
        - loss: Scalar giving the loss on the minibatch
        - grads: Dictionary mapping parameter names to gradients of the loss with
          respect to the corresponding parameter.
    - pred_func: prediction function that im
    - X: A PyTorch tensor of shape (N, D) giving training data.
    - y: A PyTorch tensor of shape (N,) giving training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - X_val: A PyTorch tensor of shape (N_val, D) giving validation data.
    - y_val: A PyTorch tensor of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.

    Returns: A dictionary giving statistics about the training process
    """
    num_train = X.shape[0] #200
    iterations_per_epoch = max(num_train // batch_size, 1) #1
    

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        # Compute loss and gradients using the current minibatch
        loss, grads = loss_func(params, X_batch, y=y_batch, reg=reg)
        if torch.isnan(loss):
            break
        loss_history.append(loss.item())

        #########################################################################
        # TODO: Use the gradients in the grads dictionary to update the         #
        # parameters of the network (stored in the dictionary self.params)      #
        # using stochastic gradient descent. You'll need to use the gradients   #
        # stored in the grads dictionary defined above.                         #
        #########################################################################
        # Replace "pass" statement with your code
        for param in params:
                weights = params[param]
                gradient = grads[param] 
                weights -= learning_rate*gradient
        #########################################################################
        #                             END OF YOUR CODE                          #
        #########################################################################

        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss.item()))

        # Every epoch, check train and val accuracy and decay learning rate.
        if it % iterations_per_epoch == 0:
            # Check accuracy
            y_train_pred = pred_func(params, loss_func, X_batch)
            train_acc = (y_train_pred == y_batch).float().mean().item()
            y_val_pred = pred_func(params, loss_func, X_val)
            val_acc = (y_val_pred == y_val).float().mean().item()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            # Decay learning rate
            learning_rate *= learning_rate_decay

    return {
        "loss_history": loss_history,
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,
    }


def nn_predict(
    params: Dict[str, torch.Tensor], loss_func: Callable, X: torch.Tensor
):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - loss_func: a loss function that computes the loss and the gradients
    - X: A PyTorch tensor of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A PyTorch tensor of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    # Replace "pass" statement with your code
    
    y_pred = torch.zeros(X.size(0), device="cuda:0")
    score = nn_forward_backward(params, X)
    y_pred = torch.argmax(score, dim=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


def nn_get_search_params():
    """
    Return candidate hyperparameters for a TwoLayerNet model.
    You should provide at least two param for each, and total grid search
    combinations should be less than 256. If not, it will take
    too much time to train on such hyperparameter combinations.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - hidden_sizes: hidden value sizes, e.g. [8, 16, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    - learning_rate_decays: learning rate decay candidates
                                e.g. [1.0, 0.95, ...]
    """
    learning_rates = []
    hidden_sizes = []
    regularization_strengths = []
    learning_rate_decays = []
    ###########################################################################
    # TODO: Add your own hyper parameter lists. This should be similar to the #
    # hyperparameters that you used for the SVM, but you may need to select   #
    # different hyperparameters to achieve good performance with the softmax  #
    # classifier.                                                             #
    ###########################################################################
    # Replace "pass" statement with your code
    '''Previous Trials
    Learning Rates: 1.0, 0.999, 0.9, 0.5, 0.3, 0.1 
    Hidden Sizes: 32, 64, 128, 256, 512, 1024
    Regularization Strengths: 0.1, 0.01, 0.0001
    Learning Rate Decays: 1, 0.999, 0.995, 0.99
    Number of Iterations: 3000, 5000, 10000, 15000
    Batch Size: 1000
    '''
    learning_rates = [1.0] 
    hidden_sizes = [256]  
    regularization_strengths = [0.0001] 
    learning_rate_decays = [0.995]
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return (
        learning_rates,
        hidden_sizes,
        regularization_strengths,
        learning_rate_decays,
    )


def find_best_net(
    data_dict: Dict[str, torch.Tensor], get_param_set_fn: Callable
):
    """
    Tune hyperparameters using the validation set.
    Store your best trained TwoLayerNet model in best_net, with the return value
    of ".train()" operation in best_stat and the validation accuracy of the
    trained best model in best_val_acc. Your hyperparameters should be received
    from in nn_get_search_params

    Inputs:
    - data_dict (dict): a dictionary that includes
                        ['X_train', 'y_train', 'X_val', 'y_val']
                        as the keys for training a classifier
    - get_param_set_fn (function): A function that provides the hyperparameters
                                   (e.g., nn_get_search_params)
                                   that gives (learning_rates, hidden_sizes,
                                   regularization_strengths, learning_rate_decays)
                                   You should get hyperparameters from
                                   get_param_set_fn.

    Returns:
    - best_net (instance): a trained TwoLayerNet instances with
                           (['X_train', 'y_train'], batch_size, learning_rate,
                           learning_rate_decay, reg)
                           for num_iter times.
    - best_stat (dict): return value of "best_net.train()" operation
    - best_val_acc (float): validation accuracy of the best_net
    """

    best_net = None
    best_stat = None
    best_val_acc = 0.0

    #############################################################################
    # TODO: Tune hyperparameters using the validation set. Store your best      #
    # trained model in best_net.                                                #
    #                                                                           #
    # To help debug your network, it may help to use visualizations similar to  #
    # the ones we used above; these visualizations will have significant        #
    # qualitative differences from the ones we saw above for the poorly tuned   #
    # network.                                                                  #
    #                                                                           #
    # Tweaking hyperparameters by hand can be fun, but you might find it useful #
    # to write code to sweep through possible combinations of hyperparameters   #
    # automatically like we did on the previous exercises.                      #
    #############################################################################
    # Replace "pass" statement with your code
    learning_rates, hidden_sizes, regularization_strengths, learning_rate_decays = get_param_set_fn()
    
    for hs in hidden_sizes:
        for lr in learning_rates:
            for rs in regularization_strengths:
                for lrd in learning_rate_decays:
                    print(f"Hyperparameters:\n- Hidden Size (hs): {hs}\n- Learning Rate (lr): {lr}\n- Regularization Strengths (rs): {rs}\n- Learning Rates Decays (lrs): {lrd}") 
                    net = TwoLayerNet(3 * 32 * 32, hs, 10, device=data_dict['X_train'].device, dtype=data_dict['X_train'].dtype)
                    stats = net.train(data_dict['X_train'], data_dict['y_train'], data_dict['X_val'], data_dict['y_val'],
                        num_iters=15000, batch_size=1000,
                        learning_rate=lr, learning_rate_decay=lrd,
                        reg=rs, verbose=True)
                    current_val_acc_history = stats["val_acc_history"]
                    print(current_val_acc_history[-1])
                    if (best_stat == None) or (current_val_acc_history[-1] > best_val_acc):
                        best_val_acc = current_val_acc_history[-1]
                        best_net = net
                        best_stat = stats
    #############################################################################
    #                               END OF YOUR CODE                            #
    #############################################################################

    return best_net, best_stat, best_val_acc
