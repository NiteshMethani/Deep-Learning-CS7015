# make
# time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp1.bin -size 50 -window 1 -sample 1e-5 -hs 0 -negative 3 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 0 -read-vocab ../corpus/banglaVocab.txt
#
# make clean
# make
# time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp2.bin -size 100 -window 1 -sample 1e-5 -hs 0 -negative 3 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 0 -read-vocab ../corpus/banglaVocab.txt
#
# make clean
# make
# time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp3.bin -size 150 -window 1 -sample 1e-5 -hs 0 -negative 3 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 0 -read-vocab ../corpus/banglaVocab.txt
#
# make clean
# make
# time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp4.bin -size 100 -window 5 -sample 1e-5 -hs 0 -negative 3 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 0 -read-vocab ../corpus/banglaVocab.txt
#
# make clean
# make
# time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp5.bin -size 100 -window 15 -sample 1e-5 -hs 0 -negative 3 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 0 -read-vocab ../corpus/banglaVocab.txt

make clean
make
time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp6.bin -size 100 -window 10 -sample 1e-5 -hs 0 -negative 3 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 0 -read-vocab ../corpus/banglaVocab.txt

make clean
make
time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp7.bin -size 100 -window 10 -sample 1e-5 -hs 0 -negative 10 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 0 -read-vocab ../corpus/banglaVocab.txt

make clean
make
time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp8.bin -size 100 -window 10 -sample 1e-5 -hs 0 -negative 25 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 0 -read-vocab ../corpus/banglaVocab.txt

# with hs
make clean
make
time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp9.bin -size 50 -window 1 -sample 1e-5 -hs 1 -negative 0 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 0 -read-vocab ../corpus/banglaVocab.txt

make clean
make
time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp10.bin -size 100 -window 1 -sample 1e-5 -hs 1 -negative 0 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 0 -read-vocab ../corpus/banglaVocab.txt

make clean
make
time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp11.bin -size 150 -window 1 -sample 1e-5 -hs 1 -negative 0 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 0 -read-vocab ../corpus/banglaVocab.txt

make clean
make
time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp12.bin -size 100 -window 5 -sample 1e-5 -hs 1 -negative 0 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 0 -read-vocab ../corpus/banglaVocab.txt

make clean
make
time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp13.bin -size 100 -window 15 -sample 1e-5 -hs 1 -negative 0 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 0 -read-vocab ../corpus/banglaVocab.txt

make clean
make
time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp14.bin -size 100 -window 10 -sample 1e-5 -hs 1 -negative 0 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 0 -read-vocab ../corpus/banglaVocab.txt

# with cbow
make clean
make
time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp15.bin -size 50 -window 1 -sample 1e-5 -hs 0 -negative 3 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 1 -read-vocab ../corpus/banglaVocab.txt

make clean
make
time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp16.bin -size 100 -window 1 -sample 1e-5 -hs 0 -negative 3 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 1 -read-vocab ../corpus/banglaVocab.txt

make clean
make
time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp17.bin -size 150 -window 1 -sample 1e-5 -hs 0 -negative 3 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 1 -read-vocab ../corpus/banglaVocab.txt

make clean
make
time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp18.bin -size 100 -window 5 -sample 1e-5 -hs 0 -negative 3 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 1 -read-vocab ../corpus/banglaVocab.txt

make clean
make
time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp19.bin -size 100 -window 15 -sample 1e-5 -hs 0 -negative 3 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 1 -read-vocab ../corpus/banglaVocab.txt

make clean
make
time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp20.bin -size 100 -window 10 -sample 1e-5 -hs 0 -negative 3 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 1 -read-vocab ../corpus/banglaVocab.txt

make clean
make
time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp21.bin -size 100 -window 10 -sample 1e-5 -hs 0 -negative 10 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 1 -read-vocab ../corpus/banglaVocab.txt

make clean
make
time ./word2vec -train ../corpus/cleanSentences.txt -output ../vectors/exp22.bin -size 100 -window 10 -sample 1e-5 -hs 0 -negative 25 -threads 20 -binary 1 -iter 15 -min-count 10 -classes 0 -cbow 1 -read-vocab ../corpus/banglaVocab.txt
