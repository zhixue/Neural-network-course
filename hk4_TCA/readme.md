<h2 style="text-align:center">Neural Network Theory and Applications  

Homework 4  

</h2>

Two problems are given below. The dataset used in this homework is the SEED dataset, which is a public EEG-based three-category emotion recognition dataset.  
EEG_X.mat: DE features for 15 different subjects.  
EEG_Y.mat: Emotion categories for 15 different subjects.  
Each subject contains 3394 samples. You need to conduct leave-one-subject out
cross validation to evaluate the performance of the algorithms. For each time, you need to choose one subject as the target subject (test set) and leave the other 14 subjects as the source subject (training set).
## Problem 1
> Solve the domain shift problem in the given dataset using support vector machines (SVMs). You need to finetune the parameters and only present the best result.

* I use one-vs-rest svm model(from libsvm). 14 subjects datasets are used for trainning, 1 subject dataset is used for checking accuary(source subject) by leaving-one-subject out
cross validation.
* In order to make the experiment repeatable, I use the 4th subject dataset as cross validation(target subject). The parameters I used are `-c 4 -b 1` which mean `cost parameter=4` and training a svc model for probability and final cross validation accuracy I got is 0.5598114319387154.
* details in `hk4.py`.


## Problem 2
> Solve the domain shift problem using TCA.  
(a) You need to implement TCA [1] and use it to solve this problem.  
(b) Compare the results with problem 1.  
(c) Alter the latent dimension of TCA and compare the results. 

a. 
* TCA is used but I only use 1st~8th subjects datasets(1st~3rd & 5th~8th as source, 4th as target), because 15 subjects datasets need a lot of memory. Use the following commands to got the new space `xproj`which is saved as `s8_X_D2.mat`(2 dimensions) and `s8_X_D10.mat`(10 dimensions) for next steps.
```bash
cd /share/matlab/R2017a/bin
./matlab -nodisplay -r "run /share/home/hzxue/domain-adaptation-toolbox/hk4.m; exit"
```
b&c. 
* It is not fair to compare TCA accuracy with the accuracy in the first problem due to different subjects are as trainning data. So I compare the accuracy with only 1~8 subjects(4th subject as target domain or cross validation).
* Accuracy:

svm | svm with tca (10 dimensions) | svm with tca (2 dimensions)
---|---|---
0.4443134944018857 | 0.3447259870359458 | 0.3447259870359458

* It seems that dimensions are not enough to support that tca has benefit because the accuracy decreases to 1/3 with X of 10 dimensions. So we can infer that the accuracy with X of 2 dimensions is **no better** than the accuracy with X of 10 dimensions. I guess that the accuracy may be a litter higher if more dimensions are included.

* details in `hk4.py` and `hk4.m`.


---
### Files description:
1. hk4.py - problem 1 & 2 svm model
2. hk4.m - tca code writen in matlab which needs function file `ftTrans_tca`, input: raw data `X` and `Y` ,output: data `xproj` after tca
3. s8_X_D2.mat - `xproj`(2 dimensions) after tca
4. s8_X_D10.mat - `xproj`(10 dimensions) after tca