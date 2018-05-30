# Run this script to execute the NN
#Note : sizes of the hidden layer neurons should be passed as space separated list: --sizes 100 100 100

python2 train.py --lr 0.001 --momentum 0.9 --num_hidden 2 --sizes 100 100 --activation tanh --loss ce --opt adam --anneal True --save_dir ../logs/dummy/ --expt_dir ../logs/dummy/ --train ../data/train.csv --val ../data/val.csv --test ../data/test.csv
