import numpy as np
import scipy.io
import sys
# add svmlib
path = '/Users/hzxue/Downloads/libsvm-3.23/python'
sys.path.append(path)
from svmutil import *


def leave_one_out(data,label,test_idx=-1):
    if test_idx == -1:
        test_idx = np.random.randint(len(label) + 1)
    cross_val_data = data[test_idx]
    cross_val_label = label[test_idx]
    temp_train_data = np.delete(data,test_idx)
    temp_train_label = np.delete(label,test_idx)
    train_data = temp_train_data[0]
    train_label = temp_train_label[0]
    for i in range(len(temp_train_label)):
        train_data = np.vstack((train_data,temp_train_data[i]))
        train_label = np.vstack((train_label,temp_train_label[i]))
    train_label = np.array([train_label[i][0] for i in range(len(train_label))])
    cross_val_label = np.array([cross_val_label[i][0] for i in range(len(cross_val_label))])
    return train_data,train_label,cross_val_data,cross_val_label


def labeltomatrix(y):
    if y[0] == -1:
        return (1,0,0)
    elif y[0] == 0:
        return (0,1,0)
    elif y[0] == 1:
        return (0,0,1)


def collect_target_label_idx(label,target):
    return tuple([i for i in range(len(label)) if label[i] == target])


def get_data_for_idx(data,idxtuple):
    return tuple([data[i] for i in idxtuple])

def one_vs_rest_train(class1_train_data,class2_train_data,class3_train_data,class1_sign,class2_sign,class3_sign): # -1 , 0 ,1
    notclass1_train_data = class2_train_data + class3_train_data
    ## model1: class1 vs not class1
    model1 = svm_train([class1_sign] * len(class1_train_data) + [-2] * len(notclass1_train_data),
                       class1_train_data + notclass1_train_data, '-c 4 -b 1')
    ## model2: class2 vs class3
    model2 = svm_train([class2_sign] * len(class2_train_data) + [class3_sign] * len(class3_train_data),
                       class2_train_data + class3_train_data, '-c 4 -b 1')
    return model1,model2


def one_vs_rest_predict(test_data,test_label,model1,model2,printOn=False):
    p1_label, p1_acc, p1_val = svm_predict(test_label,test_data,model1,'-q')

    notclass1_test_data_idx = collect_target_label_idx(p1_label,-2)
    rest_class_test_data = get_data_for_idx(test_data,notclass1_test_data_idx)
    rest_class_test_label = get_data_for_idx(test_label,notclass1_test_data_idx)

    p2_label,p2_acc, p2_val = svm_predict(rest_class_test_label,rest_class_test_data,model2,'-q')

    # merge p2_label to p1_label
    for idx1 in range(len(p1_label)):
        if p1_label[idx1] == -2:
            idx2 = notclass1_test_data_idx.index(idx1)
            p1_label[idx1] = p2_label[idx2]
            p1_val[idx1] = p2_val[idx2]

    #print(sum([1 for i in range(len(p1_label)) if p1_label[i]==-2]))

    if printOn == True:
        # compute acc
        wrong_test_num = 0
        test_num = len(test_label)
        for i in range(test_num):
            if test_label[i] != p1_label[i]:
                wrong_test_num += 1
        print("Accuracy:",str(1-wrong_test_num/test_num))
    return tuple(p1_label),p1_val


# load data
## use k data set,if use all data k = 15, use k=4 if you campare it with tca
k = 8
data = scipy.io.loadmat('EEG_X.mat')['X'][0][:k]
label = scipy.io.loadmat('EEG_Y.mat')['Y'][0][:k]

t_d,t_l,c_d,c_l = leave_one_out(data,label,test_idx=3)


## split data
classa_idx = collect_target_label_idx(t_l,-1)
classb_idx = collect_target_label_idx(t_l,0)
classc_idx = collect_target_label_idx(t_l,1)

classa_train_data = get_data_for_idx(t_d,classa_idx)
classb_train_data = get_data_for_idx(t_d,classb_idx)
classc_train_data = get_data_for_idx(t_d,classc_idx)



####################  problem 1: model one vs rest svm  ############################
print('##problem1 - start training##')
model1,model2 = one_vs_rest_train(classa_train_data,classb_train_data,classc_train_data,-1,0,1)
print('##problem1 - Train result##')
# train Accuracy: 1.0
temp = one_vs_rest_predict(t_d,t_l,model1,model2,True)
print("##problem1 - Test result##")
## all data set (k = 15) test Accuracy: 0.5598114319387154
temp = one_vs_rest_predict(c_d,c_l,model1,model2,True)


####################  problem 2: model tca(10 dimensions)  ############################
cross_idx = 3
data2 = scipy.io.loadmat('s8_X_D10.mat')['xproj']
t_d2 = np.vstack((data2[:3394*cross_idx], data2[3394*(cross_idx+1):]))
c_d2 = data2[3394*cross_idx:3394*(cross_idx+1)]
c_l2 = t_l[3394*cross_idx:3394*(cross_idx+1)]
t_l2 = np.hstack((t_l[:3394*cross_idx], t_l[3394*(cross_idx+1):]))

classa_idx = collect_target_label_idx(t_l2,-1)
classb_idx = collect_target_label_idx(t_l2,0)
classc_idx = collect_target_label_idx(t_l2,1)

classa_train_data = get_data_for_idx(t_d2,classa_idx)
classb_train_data = get_data_for_idx(t_d2,classb_idx)
classc_train_data = get_data_for_idx(t_d2,classc_idx)


print('##problem2 - start training##')
model1,model2 = one_vs_rest_train(classa_train_data,classb_train_data,classc_train_data,-1,0,1)
print('##problem2 - Train result##')
# train Accuracy: 1.0
temp = one_vs_rest_predict(t_d2,t_l2,model1,model2,True)

print("##problem2 - Test result##")
temp = one_vs_rest_predict(c_d2,c_l2,model1,model2,True)

####################  problem 2: model tca(2 dimensions)  ############################
print('##problem2 - model tca(10 dimensions)##')
cross_idx = 3
data2 = scipy.io.loadmat('s8_X_D2.mat')['xproj']
t_d2 = np.vstack((data2[:3394*cross_idx], data2[3394*(cross_idx+1):]))
c_d2 = data2[3394*cross_idx:3394*(cross_idx+1)]
c_l2 = t_l[3394*cross_idx:3394*(cross_idx+1)]
t_l2 = np.hstack((t_l[:3394*cross_idx], t_l[3394*(cross_idx+1):]))

classa_idx = collect_target_label_idx(t_l2,-1)
classb_idx = collect_target_label_idx(t_l2,0)
classc_idx = collect_target_label_idx(t_l2,1)

classa_train_data = get_data_for_idx(t_d2,classa_idx)
classb_train_data = get_data_for_idx(t_d2,classb_idx)
classc_train_data = get_data_for_idx(t_d2,classc_idx)

print('##problem2 - model tca(2 dimensions)##')
print('##problem2 - start training##')
model1,model2 = one_vs_rest_train(classa_train_data,classb_train_data,classc_train_data,-1,0,1)
print('##problem2 - Train result##')
# train Accuracy: 1.0
temp = one_vs_rest_predict(t_d2,t_l2,model1,model2,True)

print("##problem2 - Test result##")
temp = one_vs_rest_predict(c_d2,c_l2,model1,model2,True)