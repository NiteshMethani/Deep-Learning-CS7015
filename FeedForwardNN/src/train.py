from __future__ import division
import numpy as np
np.random.seed(1234)

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse
import FeedForwardNetwork as nn
import fashion_mnist_loader as loader

import seaborn as sns
from pylab import rcParams
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 12, 6

############################################################################
#Global variable definition
n_classes = 10
size = [784,n_classes]
g = ['stable_softmax']
max_epochs = 50
seed = 1234
#############################################################################

#############################################################################
#Global functions
#Commandline argument parser
def argument_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--lr', help='Learning Rate', required=True,type = float)
    parser.add_argument('--momentum',help='Momentum', default = 0.0,type = float)
    parser.add_argument('--num_hidden', help='No. of hidden layers', required=True,type=int)
    parser.add_argument('--sizes',help='Size of hidden layers', required=True,nargs='+',type=int)
    parser.add_argument('--activation', help='sigmoid/tanh', required=True)
    parser.add_argument('--loss',help='sq/ce', required = True)
    parser.add_argument('--opt', help='gd/momentum/adam/rmsprop', required=True)
    #parser.add_argument('--batch_size',help='Batch size multiples of 5', required = True,type = int)
    parser.add_argument('--anneal', help='Annealing of eta', default = False,type = bool)
    parser.add_argument('--save_dir',help='Directory name to save paramteres', required = True)
    parser.add_argument('--expt_dir',help='Directory name to save log files', required = True)
    parser.add_argument('--train',help='Training data File', required = True)
    parser.add_argument('--val',help='Validation Data File', required = True)
    parser.add_argument('--test',help='testing data File', required = True)

    args = vars(parser.parse_args())
    if (args['num_hidden'] != len(args['sizes'])) :
        print('Number of hidden layers and length of sizes parameter do not match, please try again!')

    return args

###################################################################################
def make_sure_path_exists(path):
    import os
    import errno
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
####################################################################################
#MAIN function starts here :

if __name__=='__main__':
    
    args = argument_parser()
    
    for i in args['sizes'] :
        size.insert(1,i)
    for i in range(len(args['sizes'])):
        g.insert(0,args['activation'])

    
    #Loading Data
    (X_train, Y_train, train_mean, train_std) = loader.load_data(args['train'], data = 'train', normalise = True )
    (X_val, Y_val, train_mean, train_std) = loader.load_data(args['val'], data='val', normalise = True, mean = train_mean, std = train_std)
    (id, X_test) = loader.load_test_data(args['test'], normalise = True, mean = train_mean, std = train_std)


    #Defining object and training neural network
    network = nn.FeedForwardNetwork(size, g, args['loss'], seed)
    make_sure_path_exists(args['save_dir'])
    make_sure_path_exists(args['expt_dir'])                      
    network.saveNeuralNetwork(args['save_dir'])
    batch_size = X_train.shape[0]
    (predictions, training_loss, validation_loss) = network.trainingAlgo( X_train, Y_train, X_val, Y_val, args['expt_dir'], opt = args['opt'], momentum = args['momentum'],eta=args['lr'], anneal = args['anneal'], batch_size = batch_size, max_epochs = max_epochs)
        
    
    
    df = pd.DataFrame(columns = ['training_loss', 'validation_loss'])
    df['training_loss'] = training_loss
    df['validation_loss'] = validation_loss

    df.to_csv(args['save_dir'] + 'loss.csv', index=False)
    
    
    # Making predictions on test data
    Y_test = network.forward_pass(X_test)
    Y_test = Y_test.T
    testPredictions = []
    for i in range(0, Y_test.shape[0]):
        testPredictions.append(np.argmax(Y_test[i]))
    submit = pd.DataFrame(columns = ['id','label'])
    submit['id'] = id
    submit['label'] = testPredictions
    results = submit.sort_values(by=['id'], ascending=[True])
    results.to_csv(args['save_dir'] + 'results.csv', index=False, sep=',')