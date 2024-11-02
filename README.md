Implemented a three part perceptron based regression and classification network. 

In part one, I implemented a three-layer perceptron using only numpy, implementing feed-forward and backpropagation purely mathematically. The results can be seen below:

Regression gradient descent:

![1](https://github.com/user-attachments/assets/70436a37-5a8a-4b60-927d-e78a69d45cf2)

Classification gradient descent:

![2](https://github.com/user-attachments/assets/ffbc9089-92fe-49a6-a9d1-de9e148c906c)

In part two, I implemented the same with PyTorch using the built-in error function and plotted it. The results can be seen below:

Regression gradient descent (horizontal axis is epochs and vertical axis is error):

![Figure_1](https://github.com/user-attachments/assets/c21443c8-36f0-4209-92c9-8665ce22f9a9)

Classification gradient descent (horizontal axis is epochs and vertical axis is error):

![Figure_2](https://github.com/user-attachments/assets/83a30f4c-2044-460b-9232-20806604528a)

In part three I built a grid hyperparameter search algorithm for our classification function.

I have tested hyperparameter configurations:
Number of hidden layers = [1, 2]
Number of neurons = [32, 64]
Learning rate = [0.01, 0.001]
Activation functions = [tanh, sigmoid]

For the model, I used Adam optimizer and PyTorch cross entropy function as the error function. I’ve written a flexible weight initializer and forward pass functions. Results are as follows:

![3](https://github.com/user-attachments/assets/750c2696-5789-4d36-98de-1cc8d12867c1)

More information on part three can be found in the part3_report.pdf file.

 
