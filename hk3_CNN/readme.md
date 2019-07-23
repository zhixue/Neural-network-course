<h2 style="text-align:center">Neural Network Theory and Applications  

Homework 3  

</h2>

## Problem 1
> Solving the ten-class classification problem in the given dataset using feed- forward neural network. You need to finetune your network and only present your best result.  
Notice: You can either use tensorflow or other deep learning tools to solve this problem. You can also build your network without using any deep learning tools, which is a better option.

##### Model summary
3 layers feed-forward neural network:   
* input(28 * 28 * 1),hidden layer(3000 units),output  
* activation funcition: Relu  
* learning_rate = 0.001  
* training_epochs = 1000  
* batch_size = 100

##### Result
(3 layer feed-forward) Cost trainning time = 15.84966516494751 seconds;   
Test accurary = 0.9655

## Ploblem2
> Solving the ten-class classification problem using CNN.  
(a) You need to implement LeNet[1] and use it to solve this problem.  
(b) Compare the results and training time with problem 1.  
(c) Visualize the deep features which can be extracted before feed-forward layers,
and discuss the results.  

##### Model summary
LeNet-5 has 7 layers, not counting the input.  
* C1 - a convolutional layer with 6 feature maps  
* S2 - sub-sampling layer with 6 feature maps  
* C3 - a convolutional layer with 16 feature maps  
* S4 - a sub-sampling layer with 16 feature maps  
* C5 - a convolutional layer with 120 feature maps  
* F6 - contain 84 units   
* output - 0～9
* pooling: max pooling, kernel size (2, 2), strides (2, 2), padding SAME
* convolution: weights are random got at the beginnig, strides (2, 2), padding SAME

##### Result
(LeNet5_CNN) Cost trainning time = 559.8334238529205 seconds;  
test accuracy = 0.9837

##### Performance
**(3 layer feed-forward vs LeNet5)**
model name | trainning time(seconds) | accuracy
---|---|---
3 layer feed-forward | 15.8 | 0.9655
LeNet5 | 559.8 | 0.9837

##### Deep features Visualization(only a sample)
* raw image  
 
![A58Iat.png](https://s2.ax1x.com/2019/04/08/A58Iat.png)

* first convolution  

![A58HG8.png](https://s2.ax1x.com/2019/04/08/A58HG8.png)

* first pooling  

![A58xZn.png](https://s2.ax1x.com/2019/04/08/A58xZn.png)

* second convolution  

![A5GSI0.png](https://s2.ax1x.com/2019/04/08/A5GSI0.png)

* second pooling  

![A5G9iV.png](https://s2.ax1x.com/2019/04/08/A5G9iV.png)

##### Discussion
These models have been run serveral times without large volatility. LeNet5 has a better performance than 3 layer feed-forward at the cost of more consuming time. Unusual, distorted and noisy characters correctly recognized by LeNet5.  
In these two models,`$\widehat y$` of `$ylog(\widehat y)$` in corss_entropy are processed, by using `cross_entropy = -tf.reduce_sum(Y_*tf.log(y_hat+0.0000000001))` or `cross_entropy = -tf.reduce_sum(Y_*tf.log(y_conv+0.0000000001))` in order to avoid `$log0$`.
