from utils import randInitializeWeights, Unroll_weights, Train_network
from utils import forward_prop, Mean_classification_error, cross_entropy_loss
from utils import Tanh, TanhGradient, sigmoidGradient,sigmoid, ReLu, ReLuGradient, Load_data, display_data
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    # ---------------Load & Visualize the training data------------------------
    train_file_path = './digitstrain.txt'
    val_file_path = './digitsvalid.txt'
    test_file_path = './digitstest.txt'
    X_train, Y_train = Load_data(train_file_path)
    X_val, Y_val = Load_data(val_file_path)
    X_test, Y_test = Load_data(test_file_path)
    display_data(X_train)

    #-------------------------------Network Architecture-----------------------
    input_layer_size = 784
    hidden_layer_size = [100]           # List Sizes of the hidden layer
    n_hidden = 1
    num_labels = 10

    # -------------------------Set Activation Function---------------------------
    activ_func = Tanh                   # can be sigmoid, ReLu, Tanh
    activ_Grad_func = TanhGradient      # can be sigmoidGradient, ReLuGradient, TanhGradient

    #----------------------------- Hyper Parameters -----------------------
    epochmax = 50
    LearningRate = 0.01
    reg_lambda = 0.001
    momentum = 0.1
    minibatchsize = 40 #Set =1 for regular stochastic gradient descent

    # --------------- Initializing Parameters------------------------
    num_labels = 10
    layer_sizes=[input_layer_size]
    layer_sizes.extend([hidden_layer_size[j] for j in range(len(hidden_layer_size))])
    layer_sizes.append(num_labels)
    # Initialize weights for all the layers:
    nn_weight_list = []
    for i in range(len(layer_sizes)-1):
        L_in = layer_sizes[i]
        L_out = layer_sizes[i+1]
        np.random.seed(0)
        W = randInitializeWeights(L_in, L_out) #(100, 785)
        nn_weight_list.append(W)

    """ The parameters for the neural network are "unrolled" into the vector nn_params
    and need to be converted back into the weight matrices"""
    nnparams = Unroll_weights(nn_weight_list, layer_sizes)

    # ---------------Training Network------------------------
    train_cost, val_cost, err_tr, err_val, nn_weight_list = Train_network(epochmax, reg_lambda, LearningRate,
                                        nnparams, layer_sizes, minibatchsize, momentum,  activ_func, activ_Grad_func,
                                                                          X_train, Y_train, X_val, Y_val)


    print('epochmax:{:3.0f}'.format(epochmax),' L2 Regularization: {:1.3f}'.format(reg_lambda),
  ' Learning rate: {:1.2f}'.format(LearningRate), ' Layer Sizes',layer_sizes)

    # ---------------Printing Results------------------------
    activations = forward_prop(layer_sizes, nn_weight_list, X_train, Y_train, activ_func)
    output_p = activations[-1]
    J_train= cross_entropy_loss(num_labels, output_p, Y_train, reg_lambda, nn_weight_list)
    mean_err = Mean_classification_error(Y_train,output_p)
    print 'Train  ', ' Loss: ',J_train,' Error: ', mean_err

    activation_val = forward_prop(layer_sizes, nn_weight_list, X_val, Y_val, activ_func)
    output_p = activation_val[-1]
    J_val =  cross_entropy_loss(num_labels, output_p, Y_val, reg_lambda, nn_weight_list)
    mean_err2 = Mean_classification_error(Y_val,output_p)
    print 'Validation  ','Loss: ',J_val, 'Error: ',mean_err2

    activation_test = forward_prop(layer_sizes, nn_weight_list, X_test, Y_test, activ_func)
    output_p = activation_test[-1]
    mean_err = Mean_classification_error(Y_test,output_p)
    J_test =  cross_entropy_loss(num_labels, output_p, Y_test, reg_lambda, nn_weight_list)
    print 'Test  ','Loss: ', J_test, 'Error: ',mean_err

    # ---------------Plotting Results------------------------
    print "Test Performance is", 100.0*(1-mean_err),'%'

    # Visualizing 1st layer weights:
    fig = plt.figure(figsize=(10,10))
    W = nn_weight_list[0][:,0:-1].reshape((100,28,28))
    for i in range(1,101):
        ax = plt.subplot(10,10,i)
        plt.imshow(W[i-1], cmap=plt.cm.Greys)
        ax.axis('off')
    plt.show()

    # Visualizing Loss Function and Error:
    time = np.arange(epochmax/len(train_cost),epochmax+1,epochmax/len(train_cost))
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.plot(time,train_cost,'b-.')
    plt.plot(time,val_cost,'r-.')
    plt.legend(['Train','Validation'])
    plt.title('Cross Entropy Loss vs Epoch')

    plt.subplot(122)
    plt.plot(time,err_tr,'b-.')
    plt.plot(time,err_val,'r-.')
    plt.scatter(epochmax,mean_err,c='green',marker = '+',s = 80)
    plt.legend(['Train','Validation','Test'])
    plt.title('Classification Error vs Epoch')
    plt.tight_layout()
    plt.show()
