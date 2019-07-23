import numpy as np
import sys
# add svmlib
path = '/Users/hzxue/Downloads/libsvm-3.23/python'
sys.path.append(path)
from svmutil import *


def collect_target_label_idx(label,target):
    return tuple([i for i in range(len(label)) if label[i] == target])


def get_data_for_idx(data,idxtuple):
    return tuple([data[i] for i in idxtuple])


def rand_sample(currentdata,time,idx):
    ## with no replacement
    np.random.shuffle(idx)
    left_data_idx = np.array([idx[i] for i in range(time,len(idx))])
    rand_sample_idx = np.array([idx[i] for i in range(time)])
    return tuple([currentdata[rand_sample_idx[i]] for i in range(time)]),left_data_idx


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
        correct_test_num = 0
        test_num = len(test_label)
        for i in range(test_num):
            if test_label[i] == p1_label[i]:
                correct_test_num += 1
        print("Accuracy:",str(correct_test_num/test_num))
    return tuple(p1_label),p1_val


#========= function in problem 2

def max_min_one_rest_train(class1_train_data,class2_train_data,class3_train_data,class1_sign,class2_sign,class3_sign):
    K = 2
    class23_train_data = class2_train_data + class3_train_data
    class1_idx = np.array(range(len(class1_train_data)))
    class2_idx = np.array(range(len(class2_train_data)))
    class3_idx = np.array(range(len(class3_train_data)))
    N_1_23 = min(len(class1_train_data), len(class23_train_data))
    sub_N = int(N_1_23 / K)
    models = []
    for k in range(K):
        sub_class1_train_data, class1_idx = rand_sample(class1_train_data,sub_N,class1_idx)
        sub_class2_train_data, class2_idx = rand_sample(class2_train_data, int(sub_N/2), class2_idx)
        sub_class3_train_data, class3_idx = rand_sample(class3_train_data, int(sub_N/2), class3_idx)
        models.append(one_vs_rest_train(sub_class1_train_data,sub_class2_train_data,sub_class3_train_data,class1_sign,class2_sign,class3_sign))
    return models


def max_min_part_part_train(class1_train_data,class2_train_data,class3_train_data,class1_sign,class2_sign,class3_sign):
    model_1_23 = max_min_one_rest_train(class1_train_data,class2_train_data,class3_train_data,class1_sign,class2_sign,class3_sign)
    model_2_13 = max_min_one_rest_train(class2_train_data, class1_train_data, class3_train_data, class2_sign,class1_sign, class3_sign)
    model_3_12 = max_min_one_rest_train(class3_train_data, class1_train_data, class2_train_data, class3_sign, class1_sign,class2_sign)
    return (model_1_23,model_2_13,model_3_12)







def max_tuple(lista,listb,lista_val,listb_val):
    result = [[],[]]
    for i in range(len(lista)):
        if lista_val[i] > listb_val[i]:
            result[0].append(lista[i])
            result[1].append(lista_val[i])
        else:
            result[0].append(listb[i])
            result[1].append(listb_val[i])
    return result


def min_tuple(lista,listb,lista_val,listb_val):
    result = [[], []]
    for i in range(len(lista)):
        if lista_val[i] < listb_val[i]:
            result[0].append(lista[i])
            result[1].append(lista_val[i])
        else:
            result[0].append(listb[i])
            result[1].append(listb_val[i])
    return result


def max_min_one_rest_prediect(test_data, test_label, modelK):
    pred_label = []
    ## max min######
    ## print(len(modelK))
    for i in range(len(modelK)):
        for j in range(len(modelK)):
            temp_model_plus = modelK[i][0]
            temp_model_minus = modelK[j][1]
            pred_label_temp,pred_val_temp = one_vs_rest_predict(test_data, test_label, temp_model_plus, temp_model_minus)#,True)
            pred_label.append([pred_label_temp,pred_val_temp])

    #print(l_2)
    min_pred_label_1,min_pred_label_1_val = min_tuple(pred_label[0][0],pred_label[1][0],pred_label[0][1],pred_label[1][1])
    min_pred_label_2,min_pred_label_2_val = min_tuple(pred_label[2][0],pred_label[3][0],pred_label[2][1],pred_label[3][1])
    maxmin_pred_label_val = max_tuple(min_pred_label_1,min_pred_label_2,min_pred_label_1_val,min_pred_label_2_val)
    return maxmin_pred_label_val

def vote(lista,listb,listc):
    result = []
    for i in range(len(lista[0])):
        a,b,c = lista[0][i],listb[0][i],listc[0][i]
        a_val,b_val,c_val = lista[1][i],listb[1][i],listc[1][i]
        #print(a,b,c)
        if a_val == max(a_val,b_val,c_val):
            result.append(a)
        elif b_val == max(a_val,b_val,c_val):
            result.append(b)
        else:
            result.append(c)
    return result


def max_min_part_part_prediect(test_data,test_label,modelslist,printOn=False):
    final_pred_label = []
    for modelK in modelslist:
        pred_label_temp = max_min_one_rest_prediect(test_data,test_label,modelK)
        #print(len(pred_label_temp))
        final_pred_label.append(pred_label_temp)
    final_pred_label_result = vote(final_pred_label[0],final_pred_label[1],final_pred_label[2])
    #print(len(final_pred_label_result),len(test_label))
    if printOn == True:
        # compute acc
        correct_test_num = 0
        test_num = len(test_label)
        for i in range(test_num):
            if test_label[i] == final_pred_label_result[i]:
                correct_test_num += 1
        print("Accuracy:",str(correct_test_num/test_num))
    return final_pred_label_result




## load data train: 37367*310 test: 13588*310
train_data = np.load("train_data.npy")
train_label = np.load("train_label.npy")
test_data = np.load("test_data.npy")
test_label = np.load("test_label.npy")

## split data
classa_idx = collect_target_label_idx(train_label,-1)
classb_idx = collect_target_label_idx(train_label,0)
classc_idx = collect_target_label_idx(train_label,1)

classa_train_data = get_data_for_idx(train_data,classa_idx)
classb_train_data = get_data_for_idx(train_data,classb_idx)
classc_train_data = get_data_for_idx(train_data,classc_idx)
'''
####################  problem 1: model one vs rest  ############################
print('##problem1 - start training##')
model1,model2 = one_vs_rest_train(classa_train_data,classb_train_data,classc_train_data,-1,0,1)
print('##problem1 - Train result##')
# train Accuracy: 1.0
temp = one_vs_rest_predict(train_data,train_label,model1,model2,True)
print("##problem1 - Test result##")
## test Accuracy: 0.40675596114218426
temp = one_vs_rest_predict(test_data,test_label,model1,model2,True)

################### problem 2: model max-min rest vs rest #####################
print('##problem2 - start training##')
modelKlist = max_min_part_part_train(classa_train_data,classb_train_data,classc_train_data,-1,0,1)
print("##problem2 - Train result##")
# train Accuracy: 1.0
temp = max_min_part_part_prediect(train_data,train_label,modelKlist,True)
print('##problem2 - Test result##')

temp = max_min_part_part_prediect(test_data,test_label,modelKlist,True)
'''



