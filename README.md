# MNIST_Digit_Classification

## This code implements a back propagation algorithm with cross-entropy error function for one-layer and two-layer neural networks.

#### Instructions to run the code:
1. Download Main_function.py and utils.py to  the folder containing the MNIST dataset.
2. Edit the Main_function.py file to set the Network Architecture and the hyper-parameters

## ----------------------------- Network Architecture Parameters -----------------
The Network architecture parameters are:
1. input_layer_size = 784              # for MNIST Data
2. n_hidden = 1                        # Number of Hidden Layers to use
3. hidden_layer_size = [100]           # List Sizes of the hidden layer, eg. = [100,50] for two hidden layers
4. num_labels = 10                     # Number of output classes = 10 for MNIST Data
5. activ_func = Tanh                   # Activation Function can be sigmoid, ReLu, Tanh
6. activ_Grad_func = TanhGradient      # can be sigmoidGradient, ReLuGradient, TanhGradient respectively

## ----------------------------- Hyper Parameters ---------------------------------
1. epochmax = 100                      # Maximum number of training epochs
2. LearningRate = 0.01                 # Learning rate for gradient descent parameter update
3. reg_lambda = 0.001                  # L2 Regularization Strength, weight decay
4. momentum = 0.1                      # Momentum (Keep low for high number of hidden units)
5. minibatchsize = 40                  # Set =1 for regular stochastic gradient descent, else uses Mini-batch Gradient Descent

## ----------------------------- List of Functions ---------------------------------
** utils.py **
1. Load_data(filepath='./digitstrain.txt')
2. display_data(X)
3. randInitializeWeights(L_in, L_out)
4. Unroll_weights(nn_weight_list, layer_sizes)
5. Roll_weights(nnparams,layer_sizes)
6. sigmoid(z)
7. sigmoidGradient(z)
8. softmax(x)
9. ReLu(x)
10. ReLuGradient(x)
11. Tanh(x)
12. TanhGradient(x)
13. cross_entropy_loss(num_labels, output_p, y_true, reg_lambda, nn_weight_list)
14. Mean_classification_error(Y,output_p)
15. forward_prop(layer_sizes, nn_weight_list, X, y,activ_func)
16. backprop(X, Y, activations, nn_weight_list, layer_sizes, reg_lambda, activ_Grad_func)
17. Train_network(epochmax, reg_lambda, LearningRate, nnparams, layer_sizes, minibatchsize, momentum, activ_func, activ_Grad_func, X_train, Y_train, X_val, Y_val)

