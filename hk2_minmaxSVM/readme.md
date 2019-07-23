<h2 style="text-align:center">Neural Network Theory and Applications  

Homework 2  

</h2>

## Problem 1
> Solving the three-class classification problem in the given dataset using SVM classifiers and the one-vs-rest strategy. SVM classifiers are provided in LibSVM package and other machine learning libraries (sklearn). You can use these libraries to solve this problem.

##### 解题关键策略和思想(代码见hw2_2.py):  
* svm训练和预测使用libsvm。  
* 用one-vs-rest将问题先将训练数据分成标签为 -1 和 标签为非 -1的两类问题，对应代码中的`model1`。再将非-1类分为 标签为 0 和 标签为 1 的两类问题，对应代码中的`model2`。  
* 将one-vs-rest的训练步骤和预测步骤封装成函数`one_vs_rest_train`和`one_vs_rest_predict`。若第一个分类器`model1`判断结果不为-1，则需要再经过第二个分类器`model2`判断分类结果，并更新到预测的结果中。    
* svm训练模型的参数为`-c 4 -b 1 -t 2`,即Cost参数为4，并估算正确概率

##### 代码函数功能描述：  
`collect_target_label_idx(label,target)`  
* 抽取label为指定target值的所有数据的下标并以tuple返回

`get_data_for_idx(data,idxtuple)`  
* 从一个下标的tuple获得对应的数据data并以tuple返回

`one_vs_rest_train(class1_train_data,class2_train_data,class3_train_data,class1_sign,class2_sign,class3_sign)`  
* 以标签label为class1_sign的class1作为第一个类，把标签label为class2_sign的class2和标签label为class3_sign的class3作为**非第一个类**，进行one-vs-rest，返回的是两个分类器的tuple

`one_vs_rest_predict(test_data,test_label,model1,model2,printOn=False)`
* 用one-vs-rest的一系列模型来预测输入数据的类别，输出正确率为可选项

##### 训练和测试如下：
```shell
optimization finished, #iter = 1484
nu = 0.005984
obj = -300.016891, rho = -0.298717
nSV = 272, nBSV = 81
Total nSV = 272
##problem1 - Train result##
Accuracy: 1.0
##problem1 - Test result##
Accuracy: 0.40675596114218426
```
##### 小结：
one-vs-rest的svm对于多分类效果依然不是很理想，对于训练集可能出现过拟合的情况，且训练时间长。

## Problem 2
> Solving the three-class classification problem using Min-Max- Module SVM and part-vs-part task decomposition method. You should divide the three-class problem into three two-class problems using one- vs-rest method and then decompose these imbalance two-class problems into balance two-class problems following random task decomposition and task decomposition with prior knowledge strategies. Please compare the performance of SVMs obtained in Problem one and the Min-Max- Module SVMs here.

##### 解题关键策略和思想(代码见hw2_2.py):  
1. part-vs-part的实现
* 分别构建3个one-vs-rest的两分类问题，分别让class1/class2/class3作为one的位置,其余作为rest的位置，三组结果最终通过val变量大小投票得到最终结果，体现在函数`max_min_rest_rest_train`
2. Min-Max-Module的实现
* K=2 即 每次的one-vs-part分别从class1，class2，class3数据无放回随机抽2次子集，子集之间交集为空，并集大约等于全集
* K次分别获得 `$K^+ * K^-$`个分类器（每次都有1个正类分类器和1个负类分类器），在测试阶段,将测试样本`$X$`提交给所有的子分类器`$SVM_{i,j}$`,`$i=1,..,K^{+};j=1,..,K^{-}$`,各分类器的输出为`$SVM_{i,j}(X)$`, 通过(最小)原则将得到
```math
G_{i}(X) = max_{j=1} ^{K^{-}} SVM_{i,j}(X) ,i=1,..,K^{+}
```
* 通过(最大)原则得到
```math
C(X) = max_{i=1} ^{K^{+}} G_{i}(X)
```
* 根据 `$C(X)$`的值对 `$X$` 的类别做出判断



##### 代码函数功能描述(会用到问题1的所有函数)：  
`rand_sample(currentdata,time,idx)`   
* 无放回抽取time次数据，并更新下标列表

`max_min_one_rest_train(class1_train_data,class2_train_data,class3_train_data,class1_sign,class2_sign,class3_sign)`
* 比问题1的one_rest_train多K次抽取

`max_min_part_part_train(class1_train_data,class2_train_data,class3_train_data,class1_sign,class2_sign,class3_sign)`  
* 3次one-vs-rest的实现

`max_tuple(lista,listb,lista_val,listb_val)`
* 每个位置获取大的数字  

`min_tuple(lista,listb,lista_val,listb_val)`
* 每个位置获取小的数字

`max_min_one_rest_prediect(test_data, test_label, modelK)`
* 用one-vs-rest中的K个model系列预测，用最小最大获得类别结果

`vote(lista,listb,listc)`
* 投票决定最后的类别结果（3次one-vs-rest按照最大val决定）
 
`max_min_part_part_prediect(test_data,test_label,modelslist,printOn=False)`
* 用part-vs-part的一系列模型来预测输入数据的类别，输出正确率为可选项


##### 训练和测试如下：
```shell
##problem2 - Train result##
Accuracy: 1.0
##problem2 - Test result##
Accuracy: 0.3645863997644981
```
##### 小结：
原则上Min-Max-Module SVM和part-vs-part策略应该会比one-vs-rest效果好的，但是实际没有预期的好，训练集仍有过拟合的可能，有可能是svm的参数选择有问题，或者某些细节没处理好。