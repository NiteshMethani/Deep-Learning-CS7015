# RBMs using python and numpy

In this assignment we implemented Restricted Boltzmann machines (RBMs) using only python and numpy. We were not allowed to use tensorflow, theano or any package which supports automatic differentiation (we were asked to write the backpropagation code ourselves).

### Prerequisites
```
Python 3.6.4
Numpy 1.14.0
```

### Installing

Follow these steps to get a development env running on your machine :
* Clone this repository.
* Download the FASHION MNIST dataset using the following commands and store it in code/data directory.
```
wget -P data/fashion http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget -P data/fashion http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget -P data/fashion http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget -P data/fashion http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
```

## Running the tests
* Execute `run.sh` script

## Experiments
Please find all the experiments that we conducted in the `report.pdf`.

## Authors

* **Pritha Ganguly** - [gangulypritha](https://github.com/gangulypritha)
* **Nitesh Methani** - [NiteshMethani](https://github.com/NiteshMethani)


## Acknowledgments

* This code is based on the work done by [stahp](https://github.com/basavin).
Original code can be found [here](https://github.com/basavin/rbm-smple).
