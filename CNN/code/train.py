import numpy as np
import pandas as pd
import os
import tensorflow as tf

#*******************************************************************************
def argument_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--lr', help='Learning Rate', required=True,type = float)
    parser.add_argument('--batch_size',help='Batch size multiples of 5', required = True,type = int)
    parser.add_argument('--init', help='Weight Initialization(1 for Xavier, 2 for He, 0 for random)', default = 1,type = int)
    parser.add_argument('--save_dir',help='Directory name to save paramteres', required = True)
    parser.add_argument('--train',help='Training data File', required = True)
    parser.add_argument('--val',help='Validation Data File', required = True)
    parser.add_argument('--test',help='testing data File', required = True)

    args = vars(parser.parse_args())
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

args = argument_parser()

#********************************************************************************
# UTILS

# Weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# Weight initialization (Xavier's init)
def weight_xavier_init(n_inputs, n_outputs, uniform=False):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

# Bias initialization
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 2D convolution
def conv2d(X, W):
    return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')

# Max Pooling
def max_pool_2x2(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Serve data by batches
def next_batch(batch_size):
    global index_in_epoch
    global epochs_completed
    global train_images
    global train_labels

    start = index_in_epoch
    index_in_epoch += batch_size
    num_examples = train_images.shape[0]
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch

    return train_images[start:end], train_labels[start:end]

# Convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    print num_labels
    index_offset = np.arange(num_labels) * num_classes

    dummy  = [int(x) for x in (index_offset + labels_dense.ravel())]
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[[x for x in dummy]] = 1
    return labels_one_hot


def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_everages


# Load TRAINING Data

training_file = args['train']
#training_file = './data/train.csv' # _noise_scaled_translated
print('Loding Training Data...')
data = pd.read_csv(training_file)

images = data.iloc[:,1:-1].values
images = images.astype(np.float)

# Normalize from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)
image_size = images.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

# For labels
labels_flat = data['label'].values.ravel()
labels_count = np.unique(labels_flat).shape[0]
labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)
print('Data Loaded Successfully...')
train_images = images
train_labels = labels

# LOAD VALIDATION DATA
validation_file = args['val']
#validation_file = './data/val.csv'
print('Loding Validation Data...')
data = pd.read_csv(validation_file)

images = data.iloc[:,1:-1].values
images = images.astype(np.float)

# Normalize from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)
image_size = images.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

# For labels
labels_flat = data['label'].values.ravel()
labels_count = np.unique(labels_flat).shape[0]
labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)
print('Data Loaded Successfully...')
validation_images = images
validation_labels = labels


# Parameters
n_train_samples = train_images.shape[0]
n_validation_samples = validation_images.shape[0]
EPOCHS = 15
#BATCH_SIZE = 1000
BATCH_SIZE = args['batch_size']
DISPLAY_STEP = 100
DROPOUT_CONV = 0.8
DROPOUT_HIDDEN = 0.8
VALIDATION_SIZE = n_validation_samples
TRAINING_STEPS = (n_train_samples/BATCH_SIZE)*EPOCHS

training_loss = 0.0
training_acc = 0.0
#eta = 0.0001
eta = args['lr']


# CNN MODEL

'''
Create model with 2D CNN
'''
image_size = 784
image_width = 28
image_height = 28
labels_count = 10

# Create Input and Output
X = tf.placeholder('float', shape=[None, image_size])       # mnist data image of shape 28*28=784
Y_gt = tf.placeholder('float', shape=[None, labels_count])    # 0-9 digits recognition => 10 classes
drop_conv = tf.placeholder('float')
drop_hidden = tf.placeholder('float')

# test flag for batch normalization
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
lr = tf.placeholder('float')

#**********************************************************************************
# Model Parameters
n_filters_1 = 64
n_filters_2 = 128
n_filters_3 = 256
n_filters_4 = 256
filter_size = 3
FC_1 = 1024
FC_2 = 1024

if args['init'] == 1:
    print("Xavier Initialisation")
    W1 = tf.get_variable("W1", shape=[filter_size, filter_size, 1, n_filters_1], initializer= weight_xavier_init(filter_size*filter_size*1, n_filters_1))
    W2 = tf.get_variable("W2", shape=[filter_size, filter_size, n_filters_1, n_filters_2], initializer=weight_xavier_init(filter_size*filter_size*n_filters_1, n_filters_2))
    W3 = tf.get_variable("W3", shape=[filter_size, filter_size, n_filters_2, n_filters_3], initializer=weight_xavier_init(filter_size*filter_size*n_filters_2, n_filters_3))
    W4 = tf.get_variable("W4", shape=[filter_size, filter_size, n_filters_3, n_filters_4], initializer=weight_xavier_init(filter_size*filter_size*n_filters_3, n_filters_4))
    W5_FC1 = tf.get_variable("W5_FC1", shape=[n_filters_4*4*4, FC_1], initializer=weight_xavier_init(n_filters_4*4*4, FC_1))
    W6_FC2 = tf.get_variable("W6_FC2", shape=[FC_1, FC_2], initializer=weight_xavier_init(FC_1, FC_2))
    W7_FC3 = tf.get_variable("W7_FC3", shape=[FC_2, labels_count], initializer=weight_xavier_init(FC_2, labels_count))

elif args['init'] == 2:
    print("He initialisation")
    W1 = tf.get_variable("W1", shape=[filter_size, filter_size, 1, n_filters_1], initializer= tf.contrib.layers.variance_scaling_initializer(uniform = True))
    W2 = tf.get_variable("W2", shape=[filter_size, filter_size, n_filters_1, n_filters_2], initializer=tf.contrib.layers.variance_scaling_initializer(uniform = True))
    W3 = tf.get_variable("W3", shape=[filter_size, filter_size, n_filters_2, n_filters_3], initializer=tf.contrib.layers.variance_scaling_initializer(uniform = True))
    W4 = tf.get_variable("W4", shape=[filter_size, filter_size, n_filters_3, n_filters_4], initializer=tf.contrib.layers.variance_scaling_initializer(uniform = True))
    W5_FC1 = tf.get_variable("W5_FC1", shape=[n_filters_4*4*4, FC_1], initializer=tf.contrib.layers.variance_scaling_initializer(uniform = True))
    W6_FC2 = tf.get_variable("W6_FC2", shape=[FC_1, FC_2], initializer=tf.contrib.layers.variance_scaling_initializer(uniform = True))
    W7_FC3 = tf.get_variable("W7_FC3", shape=[FC_2, labels_count], initializer=tf.contrib.layers.variance_scaling_initializer(uniform = True))

elif args['init'] == 0:
    print("Random initialisation")
    W1 = tf.get_variable("W1", shape=[filter_size, filter_size, 1, n_filters_1])
    W2 = tf.get_variable("W2", shape=[filter_size, filter_size, n_filters_1, n_filters_2])
    W3 = tf.get_variable("W3", shape=[filter_size, filter_size, n_filters_2, n_filters_3])
    W4 = tf.get_variable("W4", shape=[filter_size, filter_size, n_filters_3, n_filters_4])
    W5_FC1 = tf.get_variable("W5_FC1", shape=[n_filters_4*4*4, FC_1])
    W6_FC2 = tf.get_variable("W6_FC2", shape=[FC_1, FC_2])
    W7_FC3 = tf.get_variable("W7_FC3", shape=[FC_2, labels_count])

B1 = bias_variable([n_filters_1])
B2 = bias_variable([n_filters_2])
B3 = bias_variable([n_filters_3])
B4 = bias_variable([n_filters_4])
B5_FC1 = bias_variable([FC_1])
B6_FC2 = bias_variable([FC_2])
B7_FC3 = bias_variable([labels_count])
#**********************************************************************************

# CNN model
X1 = tf.reshape(X, [-1,image_width , image_height,1])                   # shape=(?, 28, 28, 1)

# Layer 1
l1_conv = tf.nn.relu(conv2d(X1, W1) + B1)                               # shape=(?, 28, 28, 32)
print("CONV_1 shape : {0}".format(l1_conv.shape))
# MAXPOOL
l1_pool = max_pool_2x2(l1_conv)                                        # shape=(?, 14, 14, 32)
print("POOL_1 shape : {0}".format(l1_pool.shape))
# DROPOUT
l1_drop = tf.nn.dropout(l1_pool, drop_conv)

# Layer 2
l2_conv = tf.nn.relu(conv2d(l1_drop, W2)+ B2)                           # shape=(?, 14, 14, 64)
print("CONV_2 shape : {0}".format(l2_conv.shape))
# MAXPOOL
l2_pool = max_pool_2x2(l2_conv)
print("POOL_2 shape : {0}".format(l2_pool.shape))
l2_drop = tf.nn.dropout(l2_pool, drop_conv)

# Layer 3
l3_conv = tf.nn.relu(conv2d(l2_drop, W3)+ B3)                           # shape=(?, 14, 14, 64)
print("CONV_3 shape : {0}".format(l3_conv.shape))
#l3_pool = max_pool_2x2(l3_conv)
#print("POOL_3 shape : {0}".format(l3_pool.shape))                                     # shape=(?, 7, 7, 64)
l3_drop = tf.nn.dropout(l3_conv, drop_conv)

# Layer 4
l4_conv = tf.nn.relu(conv2d(l3_drop, W4)+ B4)                           # shape=(?, 14, 14, 64)
l4_pool = max_pool_2x2(l4_conv)                                         # shape=(?, 7, 7, 64)
print("CONV_4 shape : {0}".format(l4_conv.shape))
print("POOL_4 shape : {0}".format(l4_pool.shape))
l4_drop = tf.nn.dropout(l4_pool, drop_conv)

# Layer 5 - FC1
l5_flat = tf.reshape(l4_drop, [-1, W5_FC1.get_shape().as_list()[0]])    # shape=(?, 1024)
l5_feed = tf.nn.relu(tf.matmul(l5_flat, W5_FC1)+ B5_FC1)
l5_drop = tf.nn.dropout(l5_feed, drop_hidden)

# Layer 6 - FC2
l6_feed = tf.nn.relu(tf.matmul(l5_drop, W6_FC2)+ B6_FC2)
l6_drop = tf.nn.dropout(l6_feed, drop_hidden)

# Layer 7 - FC3
l7_feed = tf.matmul(l6_drop, W7_FC3) + B7_FC3
l7_bn,update_ema1 = batchnorm(l7_feed, tst, iter, B7_FC3)
Y_pred = tf.nn.softmax(l7_bn)              # shape=(?, 10)

update_ema = tf.group(update_ema1)

# Cost function and training
cost = -tf.reduce_sum(Y_gt*tf.log(Y_pred + 1e-7))

#regularizer = (tf.nn.l2_loss(W5_FC1) + tf.nn.l2_loss(B5_FC1) + tf.nn.l2_loss(W6_FC2) + tf.nn.l2_loss(B6_FC2) + tf.nn.l2_loss(W7_FC3) + tf.nn.l2_loss(B7_FC3))
regularizer = (tf.nn.l2_loss(W7_FC3) + tf.nn.l2_loss(B7_FC3))
cost += 5e-4 * regularizer

#train_op = tf.train.AdamOptimizer(lr).minimize(cost)
train_op = tf.train.RMSPropOptimizer(lr, 0.9).minimize(cost)
correct_predict = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_gt, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))
predict = tf.argmax(Y_pred, 1)


# CREATE SESSION
'''
TensorFlow Session
'''
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# start TensorFlow session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)


train_accuracies = []
validation_accuracies = []

train_loss = []
val_loss = []

saver = tf.train.Saver()
#save_dir = 'checkpoints/'
save_dir = args['save_dir']
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')


# START SESSION
import time
start_time = time.time()

for i in range(TRAINING_STEPS):

    #get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)

     # train on batch
    _,batch_tr_loss = sess.run([train_op, cost], feed_dict={X: batch_xs, Y_gt: batch_ys, tst: False, lr :eta,drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN})
    sess.run(update_ema, feed_dict={X: batch_xs, Y_gt: batch_ys, tst: False, iter: i,lr :eta, drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN})

    training_loss = training_loss + batch_tr_loss

    batch_tr_accuracy = accuracy.eval(feed_dict={X:batch_xs,
                                                  Y_gt: batch_ys,tst: False, iter: i,
                                                  drop_conv: DROPOUT_CONV,
                                                  drop_hidden: DROPOUT_HIDDEN})
    training_acc = training_acc + batch_tr_accuracy

    # after every epoch
    if i%(num_examples/BATCH_SIZE) == 0:

        #train_loss.append(training_loss/(num_examples/BATCH_SIZE))
        train_loss.append(training_loss/(num_examples))
        train_accuracies.append(training_acc/(num_examples/BATCH_SIZE))

        validation_accuracy = accuracy.eval(feed_dict={X: validation_images,
                                                   Y_gt: validation_labels,tst: True,
                                                   drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN})
        validation_loss = sess.run(cost, feed_dict={X: validation_images, Y_gt: validation_labels, tst: True,drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN})
        val_loss.append(validation_loss/VALIDATION_SIZE)
        validation_accuracies.append(validation_accuracy)

        print("********************************************************************************")
        print('training_acc / validation_acc after epoch=> %.2f / %.2f for epoch %d'%(training_acc/(num_examples/BATCH_SIZE), validation_accuracy, int(i/(num_examples/BATCH_SIZE))))
        print('training_loss / validation_loss after epoch=> %.2f / %.2f for epoch %d'%(training_loss/(num_examples), validation_loss/VALIDATION_SIZE, int(i/(num_examples/BATCH_SIZE))))
        print("********************************************************************************")
        training_loss = 0.0
        training_acc = 0.0

    if i% DISPLAY_STEP == 0:

        validation_accuracy = accuracy.eval(feed_dict={X: validation_images,
                                                   Y_gt: validation_labels,tst: True,
                                                   drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN})
        print('validation_accuracy after %d steps => %.4f'%(i,validation_accuracy))

    #eta = eta/2.0

    saver.save(sess=sess, save_path=save_path)

print("--- %s seconds ---" % (time.time() - start_time))
# SAVE LOSSES
save_file = save_dir + 'loss.csv'
df = pd.DataFrame(columns = ['training_loss', 'validation_loss'])
df['training_loss'] = train_loss[1:]
df['validation_loss'] = val_loss[1:]
df.to_csv(save_file, index=False)

# PLOT TRAINING AND VALIDATION LOSS
import matplotlib.pyplot as plt
# %matplotlib inline
max_epochs = EPOCHS - 1
x_axis = np.linspace(0., max_epochs, num=max_epochs)

val = ['training_loss', 'validation_loss']
legend = ["Training loss", "Validation Loss"]
for idx,each in enumerate(val):
    y_axis = pd.read_csv(save_file)[each]
    #y_axis= y_axis / 5500
    plt.plot(x_axis, y_axis , label=legend[idx])
    plt.scatter(x_axis, y_axis, label=None)

plt.xlabel("# Epoch")
plt.ylabel("Loss")
plt.legend(prop={'size' : 16}) # prop is to set the font properties of the legend
plt.grid(True)
plt.title('Training vs Validation loss')
plt.savefig(save_dir + "loss.png", bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()

# check final accuracy on validation set
if(VALIDATION_SIZE):
    validation_accuracy = accuracy.eval(feed_dict={X: validation_images,
                                                   Y_gt: validation_labels,tst: True,
                                                   drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN})
    print('validation_accuracy => %.4f'%validation_accuracy)

# read test data from CSV file
test_file = args['test']
test = pd.read_csv(test_file)
test_images = test[test.columns[1:]].values
test_images = test_images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)

print('test_images({0[0]},{0[1]})'.format(test_images.shape))


# predict test set
#predicted_lables = predict.eval(feed_dict={X: test_images, keep_prob: 1.0})

# using batches is more resource efficient
predicted_lables = np.zeros(test_images.shape[0])
for i in range(0,test_images.shape[0]//BATCH_SIZE):
    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={X: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE], tst : True, drop_conv: 1.0, drop_hidden: 1.0})


# save results
np.savetxt(save_dir + 'submission.csv',
           np.c_[range(0,len(test_images)),predicted_lables],
           delimiter=',',
           header = 'id,label',
           comments = '',
           fmt='%d')

sess.close()
