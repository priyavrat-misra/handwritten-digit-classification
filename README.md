# Handwritten digit classification with Pytorch

![Cover](https://github.com/priyavrat-misra/handwritten-digit-classification/blob/master/visualizations/test_results_with_val.png?raw=true "sample test results visualization")

This project uses the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for training. It has a total of `70000` handwritten digits split into train set and test set of `60000` and `10000` images respectively. The images are __28x28 pixelated grayscale images__ of single handwritten digits between 0 and 9.

The objective of this project is to classify a given image of handwritten digit into a integer from 0 to 9.

<br>

## The process will be broken down into the following steps:
> * [Exploring the dataset](https://github.com/priyavrat-misra/handwritten-digit-classification/blob/master/data_exploration.ipynb "data_exploration.ipynb")
> * [Defining a neural network architecture](https://github.com/priyavrat-misra/handwritten-digit-classification/blob/master/network.py "network.py")
> * Hyper-parameter search and Training the model
>    - [without validation](https://github.com/priyavrat-misra/handwritten-digit-classification/blob/master/train.ipynb "train.ipynb")
>    - [with validation](https://github.com/priyavrat-misra/handwritten-digit-classification/blob/master/train_with_validation.ipynb "train_with_validation.ipynb")
> * [Evaluating the model's results and plotting graphs](https://github.com/priyavrat-misra/handwritten-digit-classification/blob/master/results.ipynb "results.ipynb")
> * Experiment how RNNs do on image data
>    - [defining various RNN architectures](https://github.com/priyavrat-misra/handwritten-digit-classification/blob/master/rnns.py "rnns.py")
>    - [training those models](https://github.com/priyavrat-misra/handwritten-digit-classification/blob/master/train_with_rnns.ipynb "train_with_rnns.ipynb")


## Results:
> || Train Accuracy | Validation Accuracy | Test Accuracy |
> | :- | -: | -: | -: |
> | `Training without validation | 99.30% | - | 99.19% |
> | `Training with validation | *99.34% | 99.06% | 99.14% |
> | ^Training with Vanilla RNN | *95.18% | - | 95.86% |
> | ^Training with GRU | *99.42% | - | 98.97% |
> | ^Training with LSTM | *99.24% | - | 98.85% |
> | ^Training with Bidirectional LSTM | *99.16% | - | 98.89% |

_<sub>* - running accuracy;</sub>_
_<sub>` - trained for 4 epochs;</sub>_
_<sub>^ - trained for 8 epochs;</sub>_

<br>

## Todo
- [x] data exploration
- [x] train model with validation
- [x] experiment how RNNs do on image data
- [ ] add data augmentation
- [ ] deploy with flask

