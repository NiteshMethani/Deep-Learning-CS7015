from __future__ import division
import numpy as np
import pandas as pd
import helper
from scipy.special import expit
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error

class FeedForwardNetwork(object):
    #
    # Architecture of the Neural Network
    # I am considering input layer as 0th layer of the neural network
    #
    n_layers  = 0
    n_neurons = []
    weights = []
    biases = []
    activation_functions = []
    layerInput = []
    layerOutput = []
    loss_function = ''
    
    #
    # Initialise the weights and biases randomly
    #
    def __init__(self, size, g, loss_function = 'ce', seed=1234):
        
        np.random.seed(seed)
        
        self.n_layers = len(size)
        
        self.n_neurons = size
        
        self.activation_functions = g
        
        # you should have activation function for each HIDDEN LAYER i.e n_layers - 1
        assert len(self.activation_functions) == self.n_layers - 1, "Number of Activation Functions should be equal to number of hidden layers"
        
        self.weights = [np.random.randn(y, x)* 2/np.sqrt(x+y) for x,y in zip(self.n_neurons[:-1], self.n_neurons[1:])]
        #for x, y in zip(self.n_neurons[:-1], self.n_neurons[1:]):
        #    w = np.random.randn(y, x)*np.sqrt(2 / (x+y)) 
        #    self.weights.append(w)

        self.biases = [np.zeros((x, 1)) for x in self.n_neurons[1:]]
        self.loss_function = loss_function

    
    #
    # Save weights
    #
    def save_weights(self, filepath):
        df = pd.DataFrame(data=self.weights)
        df.to_csv(filepath, sep=' ', header=False, float_format='%.5f', index=False)
    
    #
    # Save biases
    #
    def save_biases(self, filepath):
        df = pd.DataFrame(data=self.biases)
        df.to_csv(filepath, sep=' ', header=False, float_format='%.5f', index=False)
    
    #
    # Save the architecture of the network in a file
    #
    def saveNeuralNetwork(self, filepath):
        with open(filepath + 'Network.txt', 'w') as the_file:
            the_file.write('Feedforward Neural Network\n')
            
            the_file.write('---------------------------------------------------------------------------')
            
            the_file.write("\nNumber of Layers : %d" % (self.n_layers))
            
            the_file.write("\nNumber of neurons in each layer : ")
            for i in self.n_neurons:
                the_file.write("%d " % (i))
            
            the_file.write('\nActivation Functions used at each layer:')
            for i in self.activation_functions[:-1]:
                the_file.write("%s " % (i))
            
            the_file.write('\nOutput Function used : %s' % self.activation_functions[-1])
            
            the_file.write('\nLoss Function used : %s' % self.loss_function)
            
            the_file.write('\nWeigths and Biases are stored in initial_weights.csv and initial_biases.csv')
            
            self.save_weights(filepath + 'initial_weights.csv')
            
            self.save_biases(filepath + 'initial_biases.csv')
            
        return 'Neural Network saved successfully...'
    
    #
    # Transfer Functions
    #
    def sigmoid(self, x, derivative = False):
        if derivative == False:
            #return 1 / ( 1 + np.exp(-x))
            return expit(x)
        else:
            sigma_x = self.sigmoid(x)
            return sigma_x * (1 - sigma_x)
        
    def tanh(self, x, derivative = False):
        if derivative == False:
            return np.tanh(x)
        else:
            tanh_x = self.tanh(x)
            return (1 - np.square(tanh_x))
    
    def relu(self, x, derivative = False):
        if derivative == False:
            return np.maximum(0, x, x)
        else:
            return np.greater(x,0).astype(int)
        
    
    def softmax(self, x, derivative = False):
        if derivative == False:
            
            out = np.zeros(x.shape)
            
            for i in range(0, x.shape[1]):
                exps = np.exp(x[:, i])
                out[:, i] = exps / np.sum( exps)
                
            return out
            #return exps / np.sum(exps)
        else:
            pass
        
        return
    
    # Ref :- https://deepnotes.io/softmax-crossentropy
    def stable_softmax(self, x, derivative = False):
        if derivative == False:
            out = np.zeros(x.shape)
            for i in range(0, x.shape[1]):
                exps = np.exp(x[:, i] - np.max(x[:, i]))
                out[:, i] = exps / np.sum( exps)
                
            return out
            #return exps / np.sum(exps)
        else:
            pass
        
        return
    
        
    #
    # Training the Neural Network
    #
    def forward_pass(self, X):
    
        self.layerInput = []
        self.layerOutput = []

        W = self.weights
        bias = self.biases
        
        n_samples = X.shape[0]
        dim = X.shape[1]
        Z = []
        A = []

        # here we don't have tp consider input layer
        for layer in range(0,self.n_layers - 1):

            # Pre-activation
            
            # Check Dimensions
            assert W[layer].shape == (self.n_neurons[layer+1], self.n_neurons[layer]), 'Check Dimensions of Weights'
            
            assert bias[layer].shape == (self.n_neurons[layer+1], 1), 'Check Dimensions of Biases'
            
#             assert W[0].shape == (self.n_neurons[1],self.n_neurons[0]), "Check W[0]"
#             assert bias[0].shape == (self.n_neurons[1], 1), "Check bias[0]"
           
            assert X.shape == (n_samples,dim), "Check Input dimensions"
            
            if layer == 0:
                Z.append(np.matmul(W[layer], X.T) + bias[layer])
                # Check Dimensions
                assert Z[0].shape == (self.n_neurons[1], n_samples), "Check Z[0]"
            else:
                Z.append(np.matmul(W[layer], A[layer - 1]) + bias[layer])
                # Check Dimensions
                assert Z[layer].shape == (self.n_neurons[layer + 1], n_samples), "Check Z[layer]"
            
            # Cache Z's
            self.layerInput.append(Z[layer])

            # Activation
            if self.activation_functions[layer] == 'sigmoid':
                A.append(self.sigmoid(Z[layer]))
            elif self.activation_functions[layer] == 'tanh':
                A.append(self.tanh(Z[layer]))
            elif (self.activation_functions[layer] == 'stable_softmax') and (layer == self.n_layers - 2) :
                A.append(self.stable_softmax(Z[layer]))
            elif self.activation_functions[layer] == 'relu':
                A.append(self.relu(Z[layer]))
            
            
            # Check Dimensions
            assert A[layer].shape == Z[layer].shape, "Check A[layer]"

            # Cache A's
            self.layerOutput.append(A[layer])

         # TODO : Check Dimensions
        return self.layerOutput[-1]

 
    #
    # backward_pass()
    # TODO : predictions is not needed
    #
    def backward_pass(self, X_train, Y_train, predictions):
        grad_layers  = [0]*(self.n_layers - 1)
        grad_weights = [0]*(len(self.weights))
        grad_biases  = [0]*(len(self.biases))
        n_samples = X_train.shape[0]

        for layer in reversed(range(self.n_layers-1)):
                        
            # -2 will make the indexing of layerInput/Output and layer indexing (remember you are counting input layer in n_layers) same
            if layer == self.n_layers - 2:
                # You are at the output layer now
                # gradients with respect to the output layer
                #
                if self.loss_function == 'ce':
                    grad_layers[layer] = self.layerOutput[layer] - Y_train.T
                elif self.loss_function == 'sq':
                    grad_layers[layer] = (self.layerOutput[layer] - Y_train.T) * self.layerOutput[layer] * (1 - self.layerOutput[layer])
                
                assert grad_layers[layer].shape == self.layerInput[layer].shape, "Check dZ[2]"
                assert grad_layers[layer].shape == (self.n_neurons[layer+1], n_samples), "Check dZ[2]"

                grad_weights[layer] = (1/n_samples) * np.matmul(grad_layers[layer], self.layerOutput[layer-1].T )
                assert grad_weights[layer].shape == self.weights[layer].shape, "Check dW[2]"

                grad_biases[layer] = (1/n_samples) * np.sum(grad_layers[layer], axis=1, keepdims = True)
                assert grad_biases[layer].shape == self.biases[layer].shape, "Check dB[2]"
                
            else:
                if self.activation_functions[layer] == 'sigmoid':
                    grad_layers[layer] =(np.matmul(self.weights[layer+1].T, grad_layers[layer+1]))*(self.sigmoid(grad_layers[layer], derivative = True))
                elif self.activation_functions[layer] == 'tanh':
                    grad_layers[layer] =(np.matmul(self.weights[layer+1].T, grad_layers[layer+1]))*(self.tanh(grad_layers[layer], derivative = True))
                elif self.activation_functions[layer] == 'relu':
                    grad_layers[layer] =(np.matmul(self.weights[layer+1].T, grad_layers[layer+1]))*(self.relu(grad_layers[layer], derivative = True))

                
                # Check Dimensions
                assert grad_layers[layer].shape == self.layerInput[layer].shape, "Check dZ[1]"
                assert grad_layers[layer].shape == (self.n_neurons[layer+1], n_samples), "Check dZ[1]"
                
                if layer == 0:
                    grad_weights[layer] = (1/n_samples) * np.matmul(grad_layers[layer], X_train)
                else:
                    grad_weights[layer] = (1/n_samples) * np.matmul(grad_layers[layer], self.layerOutput[layer-1].T)
                # Check Dimensions
                assert grad_weights[layer].shape == self.weights[layer].shape, "Check dW[2]"

                grad_biases[layer] = (1/n_samples) * np.sum(grad_layers[layer], axis=1, keepdims = True)
                # Check Dimensions
                assert grad_biases[layer].shape == self.biases[layer].shape, "Check dB[2]"
            
            
        return (grad_layers, grad_biases, grad_weights)
    
    #
    # Optimization Algos = [ gd, momentum, nag, adam ]
    #
    def updateRule(grad_weights, grad_biases, eta, momentum = 0.9, beta = 0.999, epsilon = 0.00000001, opt='adam'):
        pass
        

    #
    # Training Algorithm
    #
    def trainingAlgo(self, X_train, Y_train, X_val, Y_val, filepath, opt = 'adam', momentum = 0.9, eta=0.01, anneal = False, batch_size = 20, max_epochs=100):
        eta_0 = eta
         
        log_train = open(filepath +'log_train.txt', 'a+')
        log_val = open(filepath +'log_val.txt', 'a+')
              
        prev_weights = [np.zeros((y, x)) for x,y in zip(self.n_neurons[:-1], self.n_neurons[1:])]
        prev_biases = [np.zeros((x, 1)) for x in self.n_neurons[1:]]
        
        # Check
        assert len(prev_weights) == len(self.weights), 'Check previous weights dimensions'
        assert len(prev_biases) == len(self.biases), 'Check previous biases dimensions'      
        
        training_loss = []
        validation_loss= []
        
        t = 0
        for epoch in range(max_epochs):
            step = 0           
            batch_loss = 0
            
            for num in range(0, X_train.shape[0], batch_size):
                
                # MiniBatch Loop starts here
                
                
                X_train_mini = X_train[num:num+batch_size]
                Y_train_mini = Y_train[num:num+batch_size]
                
                assert X_train_mini.shape == (batch_size, X_train.shape[1]), "Check mini-batch diensions"
                assert Y_train_mini.shape == (batch_size, Y_train.shape[1]), "Check mini-batch diensions"
                
                #
                # forward_pass()
                # predictions means predictions_batch
                #
                
                predictions = self.forward_pass(X_train_mini)
                
                #
                # backward_pass()
                #
                (grad_layers, grad_biases, grad_weights) = self.backward_pass(X_train_mini, Y_train_mini, predictions)
                
                #
                # update_Rule()
                #
                if opt == 'gd':
                    update_w = np.multiply(eta , grad_weights)
                    update_b = np.multiply(eta , grad_biases)

                elif opt == 'momentum':              
                    update_w = np.multiply(momentum, prev_weights) + np.multiply(eta, grad_weights)
                    update_b = np.multiply(momentum, prev_biases) + np.multiply(eta, grad_biases)

                    prev_weights = update_w
                    prev_biases = update_b
                
                elif opt == 'rmsprop':
                    eps, beta = 1e-8, 0.999
                    v_w, v_b = 0.0, 0.0
                    
                    v_w = np.multiply(beta,v_w) + np.multiply(1-beta,np.power(grad_weights,2))
                    v_b = np.multiply(beta,v_b)  + np.multiply(1-beta,np.power(grad_biases,2))
                    
                    v_w_corrected = 1 / (np.power(v_w + eps, 1/2))
                    v_b_corrected = 1 / (np.power(v_b + eps, 1/2))
                                         
                    update_w = np.multiply(v_w_corrected, grad_weights) * eta
                    update_b = np.multiply(v_b_corrected, grad_biases) * eta
                    
                    assert len(v_w) == len(self.weights), 'Check RMSProp code'
                    assert v_w[0].shape == self.weights[0].shape, 'Check RMSProp code'
                    
                elif opt == 'adam':
                    m_w,m_b,v_w,v_b,eps,beta1,beta2 = 0,0,0,0,1e-8,0.9,0.999
                    
                    m_w = np.multiply(beta1,m_w) + np.multiply(1-beta1,grad_weights)
                    m_b = np.multiply(beta1,m_b) + np.multiply(1-beta1,grad_biases)

                    v_w = np.multiply(beta2,v_w) + np.multiply(1-beta2,np.power(grad_weights,2))
                    v_b = np.multiply(beta2,v_b) + np.multiply(1-beta2,np.power(grad_biases,2))

                    m_w = m_w/(1 - np.power(beta1,t+1))
                    m_b = m_b/(1 - np.power(beta1,t+1))
                
                    v_w = v_w/(1 - np.power(beta2,t+1))
                    v_b = v_b/(1 - np.power(beta2,t+1))
                                        
                    update_w = (eta / np.power(v_w + eps, 1/2)) * m_w
                    update_b = (eta / np.power(v_b + eps, 1/2)) * m_b
                
                t = t+1
                self.weights = self.weights - update_w
                self.biases = self.biases - update_b
                
                #
                # Anneal
                #
                if anneal == True:
                    eta = ((0.99)**(epoch+3.5))*eta_0
                    #eta = eta / 2
                    
                step = step + 1
                
                
                
                ###############################################
                
                if self.loss_function == 'ce':           
                    batch_loss += log_loss(Y_train_mini.T,predictions)
                elif self.loss_function == 'sq':
                    batch_loss += mean_squared_error(Y_train_mini.T,predictions)
                    
                
                # End Minibatch Loop
                ###############################################
         
            #
            # Save Parameters (if you want)
            #
            paramPath = filepath+'/parameters/'
            helper.make_sure_path_exists(paramPath)
            
            if epoch == max_epochs - 1:
                self.save_weights(paramPath + 'weights' + '.csv')
                self.save_biases(paramPath +'biases' + '.csv')
            
            ##############################################################################################
            
            training_loss.append(batch_loss)
            
            predictions_train = self.forward_pass(X_train)
            (acc_train, correct_train,total_train) = self.evaluate(predictions_train.T, Y_train)
            
            print('Epoch {0} Training Loss {1}'.format(epoch, training_loss[epoch]))
            
            log_train.write('Epoch : %i Loss : %2.2f Error : %2.2f lr : %3.3f \n' % (epoch, training_loss[epoch], round(100 - acc_train , 2) , eta))
            log_train.write('--------------------------------------------------------------------------------------------------\n')
                
            
            predictions_val = self.forward_pass(X_val)
            # Loss
            if self.loss_function == 'ce':
                validation_loss.append(log_loss(Y_val.T, predictions_val))
            elif self.loss_function == 'sq': 
                validation_loss.append(mean_squared_error(Y_val.T, predictions_val))

            print('Epoch {0} Validation Loss {1}'.format(epoch, validation_loss[epoch]))
            # Error
            (acc_val, correct_val,total_val) = self.evaluate(predictions_val.T, Y_val)
            
            log_val.write('Epoch : %i Loss : %2.2f Error : %2.2f lr : %3.3f \n' % (epoch, validation_loss[epoch], round(100 - acc_val, 2), eta))
            log_val.write('--------------------------------------------------------------------------------------------------\n')
            
            #print('Epoch {0} : Training error : {1}%'.format(epoch, round(100 - acc_train , 2)))
            #print('Epoch {0} : Validation error : {1}%'.format(epoch, round(100 - acc_val , 2)))
            #print('--------------------------------------------------------------------------------------------------\n')
            
            print('Epoch {0} : Training Accuracy : {1}%'.format(epoch, round(acc_train , 2)))
            print('Epoch {0} : Validation Accuracy : {1}%'.format(epoch, round(acc_val , 2)))
            print('--------------------------------------------------------------------------------------------------\n')
            ##############################################################################################
            # End Epoch Loop
            
        log_train.close()
        log_val.close()
        
        return (predictions.T, training_loss, validation_loss)
    
    
    def evaluate(self, predictions, actuals):
        assert predictions.shape == actuals.shape, 'Check dimensions of y_hat and y'
        accuracyMatrix = np.argmax(predictions, axis = 1).reshape(actuals.shape[0], 1) == np.argmax(actuals, axis = 1).reshape(actuals.shape[0], 1)
        accuracyList = accuracyMatrix.tolist()
        correct = accuracyList.count([True])
        total = len(accuracyList)
        acc = correct/total
        acc = round(acc*100, 2)
        #print('Accuracy on Training Data : {0}/{1}, {2} % '.format(correct, total, acc))
        return (acc,correct,total)

