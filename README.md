# Handwritten digit classification with Pytorch

![Cover](https://github.com/priyavrat-misra/handwritten-digit-classification/blob/master/visualizations/test_results_with_val.png?raw=true "sample test results visualization")

This project uses the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for training. It has a total of `70000` handwritten digits split into train set and test set of `60000` and `10000` images respectively. The images are __28x28 pixelated grayscale images__ of single handwritten digits between 0 and 9.

The objective of this project is to classify a given image of handwritten digit into a integer from 0 to 9.

<br>

## Getting Started
```bash
git clone https://github.com/priyavrat-misra/handwritten-digit-classification.git
cd handwritten-digit-classification/
pip install -r requirements.txt
```

## The process will be broken down into the following steps:
> * [Exploring the dataset](https://github.com/priyavrat-misra/handwritten-digit-classification/blob/master/data_exploration.ipynb "data_exploration.ipynb")
> * [Defining a neural network architecture](https://github.com/priyavrat-misra/handwritten-digit-classification/blob/master/network.py "network.py")
> * Training the model
>    - [without validation](https://github.com/priyavrat-misra/handwritten-digit-classification/blob/master/train.ipynb "train.ipynb")
>    - [with validation](https://github.com/priyavrat-misra/handwritten-digit-classification/blob/master/train_with_validation.ipynb "train_with_validation.ipynb")
> * [Evaluating the model's results and making cool graphs!](https://github.com/priyavrat-misra/handwritten-digit-classification/blob/master/results.ipynb "results.ipynb")


## Results:
```
* Training without validation
    - Train Accuracy: 99.30%
    - Test Accuracy: 99.19%

* Training with validation
    - Train Accuracy: 99.51%
    - Test Accuracy: 99.14%
```

<br>

__Note:__ this project uses Tensorboard as an evaluation utility for checking running losses, accuracies, histograms etc. So if you are wondering why there are no outputs (running loss, epoch number etc) while the network is training, use Tensorboard (_open terminal, change path to project's repo and run this command `tensorboard --logdir=runs`_) or if you aren't a fan of 3rd party tools, adding a bunch of print statements will do the job.

<br>

## Todo
- [x] data exploration
- [x] train model with validation
- [ ] add data augmentation
- [ ] deploy with flask

