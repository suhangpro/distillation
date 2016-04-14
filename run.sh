#!/bin/sh

## Experiment 1 ##

# train on mnist550 -- (** 5.8% / 5.9% / 7.3% / 7.0% **) 
python cnn_mnist.py --dataset=mnist550 --num_epochs=400


## Experiment 2 ##

# step 1: train on letters -- 13.6% / 14.2% / 14.2% / 13.9%
python cnn_mnist.py --dataset=letters --num_epochs=20 --batch_size=128 --val_size=5000 \
--save_model_to=letters.model

# step 2: fine-tune on mnist550 -- (** 2.4% / 2.8% / 3.1% / 2.9% **)
python cnn_mnist.py --dataset=mnist550 --num_epochs=400 \
--load_model_from=letters.model


## Experiment 3 ##

# step 1: train on mnist550+letters and get logits -- 11.4% / 11.6% / 18.8% / 21.3%
python cnn_mnist.py --dataset=mnist550+letters --num_epochs=20 --batch_size=128 --val_size=5000 \
--save_logit_to=mnist550+letters.logit

# step 2: train on mnist550 with logits -- (** 5.4% / 8.2% / 7.9% / 7.6% **)
python cnn_mnist.py --dataset=mnist550 --target=logit --init_lr=0.001 --num_epochs=400 \
--load_logit_from=mnist550+letters.logit.npy


## Experiment 4 ##

# step 1: train on letters
# re-use "letters.model" from Experiment 2 / step 1

# step 2: train on mnist550+letters and get logits
# re-use "mnist550+letters.logit.npy" from Experiment 3 / step 1

# step 3: fine-tune on mnist550 with logits -- (** 1.5% / 3.4% / 4.7% / 4.5% **)
python cnn_mnist.py --dataset=mnist550 --target=logit --init_lr=0.001 --num_epochs=400 \
--load_logit_from=mnist550+letters.logit.npy --load_model_from=letters.model 

