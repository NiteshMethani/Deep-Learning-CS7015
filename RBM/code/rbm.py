from loader import load_mnist, save_mnist_image
from time import time
import numpy as np
import os
import sys


def rbm(dataset, num_hidden, learn_rate, epochs, k,batchsize):

   num_visible = dataset.shape[1]
   num_examples = dataset.shape[0]

   print("Training RBM with", num_visible, "visible units,",
          num_hidden, "hidden units,", num_examples, "examples and",
          epochs, "epochs...")

   start_time = time()

   batches = num_examples // batchsize

   w = 0.1 * np.random.randn(num_visible, num_hidden)
   a = np.zeros((1, num_visible))
   b = -4.0 * np.ones((1, num_hidden))

   # w = np.load('./Output_weights/w_v60000_h50.npy')
   # a = np.load('./Output_weights/a_v60000_h50.npy')
   # b = np.load('./Output_weights/b_v60000_h50.npy')

   w_inc = np.zeros((num_visible, num_hidden))
   a_inc = np.zeros((1, num_visible))
   b_inc = np.zeros((1, num_hidden))


   #  #Nitesh
   # print("*******************************************")
   # print('Original Data {}'.format(dataset.shape)) # 1 bacth
   # print("*******************************************")
   save_error = []
   for epoch in range(epochs):
      error = 0
      for batch in range(batches):
         #### --- Positive phase of contrastive divergence --- ####

         # get next batch of data
         v0 = dataset[int(batch*batchsize):int((batch+1)*batchsize)]


         # #Nitesh
         # print("*******************************************")
         # print('Batch Shape {}'.format(v0.shape)) # 1 bacth
         # print("*******************************************")

         # in this matrix, m[i,j] is prob h[j] = 1 given example v[i]
         # dims: [num_ex] x [num_hidden]
         for i in range(k) :
         	prob_h0 = logistic(v0, w, b)

         	# sample the states of hidden units based on prob_h0
         	h0 = prob_h0 > np.random.rand(batchsize, num_hidden)

         	# reconstruct the data by sampling the visible states from hidden states
         	v1 = logistic(h0, w.T, a)

         	# sample hidden states from visible states
         	prob_h1 = logistic(v1, w, b)
         	#print("Completed one round of k")

         # positive phase products
         vh0 = np.dot(v0.T, prob_h0)

         # activation values needed to update biases
         poshidact = np.sum(prob_h0, axis=0)
         posvisact = np.sum(v0, axis=0)

         #### --- Negative phase of contrastive divergence --- ####


         #negative phase products
         vh1 = np.dot(v1.T, prob_h1)

         # activation values needed to update biases
         neghidact = np.sum(prob_h1, axis=0)
         negvisact = np.sum(v1, axis=0)

         #### --- Updating the weights --- ####

         # set momentum as per Hinton's practical guide to training RBMs
         m = 0.5 if epoch > 5 else 0.9

         # update the weights
         w_inc = w_inc * m + (learn_rate/batchsize) * (vh0 - vh1)
         a_inc = a_inc * m + (learn_rate/batchsize) * (posvisact - negvisact)
         b_inc = b_inc * m + (learn_rate/batchsize) * (poshidact - neghidact)

         a += a_inc
         b += b_inc
         w += w_inc

         error += np.sum((v0 - v1) ** 2)
      save_error.append(error/60000)
      print("Epoch %s completed. Reconstruction error is %0.2f. Time elapsed (sec): %0.2f. lr= %0.7f"
            % (epoch + 1, error/60000, time() - start_time, learn_rate))

   print ("Training completed.\nTotal training time (sec): %0.2f \n" % (time() - start_time))
   np.savetxt('loss_h_'+str(num_hidden)+'_k_'+str(k)+'.txt', save_error)
   return w, a, b


def logistic(x,w,b):
   xw = np.dot(x, w)
   replicated_b = np.tile(b, (x.shape[0], 1))

   return 1.0 / (1 + np.exp(- xw - b))


def reconstruct(v0, w, a, b):
   num_hidden = w.shape[1]
   prob_h0 = logistic(v0, w, b)
   h0 = prob_h0 > np.random.rand(1, num_hidden)

   return logistic(h0, w.T, a)

def sample_hidden(v0,w,b):

   num_hidden = w.shape[1]
   return logistic(v0, w, b)


def save_weights(w, a, b, directory, n_examples, num_hidden):

   if not os.path.exists(directory):
      os.makedirs(directory)

   w_name = directory + os.sep + "w_v" + str(n_examples) + "_h" + str(num_hidden)
   np.save(w_name, w)
   a_name = directory + os.sep + "a_v" + str(n_examples) + "_h" + str(num_hidden)
   np.save(a_name, a)
   b_name = directory + os.sep + "b_v" + str(n_examples) + "_h" + str(num_hidden)
   np.save(b_name, b)


def test_mnist(n_examples, num_hidden, epochs, learn_rate,k):

   # load data
   images, labels = load_mnist(n_examples, training = True)

   # train one layer of RBM
   w, a, b  = rbm(images, num_hidden, learn_rate, epochs, k,batchsize = 30)

   # Load weights #NITESH
   # w = np.load('./Output_weights/w_v60000_h300.npy')
   # a = np.load('./Output_weights/a_v60000_h300.npy')
   # b = np.load('./Output_weights/b_v60000_h300.npy')
   # save all weights
   print("Saving weights...")
   save_weights(w, a, b, "Output", n_examples, num_hidden)

   # try to reconstruct some test set images
   print("Generating and saving the reconstructed images...")
   samples = 60000
   images, labels = load_mnist(samples, training = True)
   visible_rep = np.zeros((samples, 784))
   visible_label = np.zeros((samples,1))

   hidden_rep = np.zeros((samples, num_hidden))
   hidden_label = np.zeros((samples,1))
   i=59001
   data = images[i]
   save_mnist_image(data, "Nitesh", str(i) + "original.png")
   for i in range(samples):
      data = images[i]
      save_mnist_image(data, "Output", str(i) + "original.png")
      data1 = reconstruct(data, w, a, b)
      visible_rep[i] = data1
      visible_label[i] = labels[i]
      data2 = sample_hidden(data, w, b)
      hidden_rep[i] = data2
      hidden_label[i] = labels[i]
      save_mnist_image(data1, "Output", str(i) + "reconstructed_visible.png")
      # NITESH
      #save_mnist_image(data2, "Output", str(i) + "reconstructed_hidden.png", hidden=True,num_hidden=num_hidden)
   print("Done!")

    #Nitesh
   # print("*******************************************")
   # print('Reconstructed Images {}'.format(data.shape)) # 1 bacth
   # print("*******************************************")
   np.savetxt('visible/visible_representation_n_h_'+str(num_hidden)+'_k_'+str(k)+'.txt', visible_rep)
   np.savetxt('visible/visible_representation_labels_n_h_'+str(num_hidden)+'_k_'+str(k)+'.txt', visible_label)

   np.savetxt('hidden/hidden_representation_nh_'+str(num_hidden)+'_k_'+str(k)+'.txt', hidden_rep)
   np.savetxt('hidden/hidden_representation_labels_nh_'+str(num_hidden)+'_k_'+str(k)+'.txt', hidden_label)

if __name__ == '__main__':
   if len(sys.argv) == 6:
      # user provided the params
      n_examples = int(sys.argv[1])
      num_hidden = int(sys.argv[2])
      epochs = int(sys.argv[3])
      learn_rate = float(sys.argv[4])
      k = int(sys.argv[5])

      print('Number of Hidden {}'.format(num_hidden))
      # run!
      test_mnist(n_examples, num_hidden, epochs, learn_rate,k)
   else:
      test_mnist(60000, 100, 1000, 0.001,1)
