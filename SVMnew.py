# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 16:20:27 2018

@author: Kripa
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import time

row=['Id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type']
li = pd.read_csv('glass.data',header=None,names=row)
li=li.drop(columns=['Id'])
trainX = np.zeros((214,10))
trainY = np.zeros((214,))
trainX= np.array(li)[:,:]
np.random.shuffle(trainX) #To shuffle the input data 

train = np.zeros((214,9))
trainlabel= np.zeros((214,1))
train = np.array(trainX)[:,:-1] #To create an array after dropping type and id
trainlabel=np.array(trainX)[:,9]

#Function for Linear kernel
def svc_linear(Cs,test_label1,test_set1,training_label1,training_set1,dfs,weight=None):
    Accuracy = 0
    param = {i:[]}
    start=time.time()
    for c in Cs:
        clf = svm.SVC(kernel=k, C=c,decision_function_shape = dfs,class_weight=weight).fit(training_set1, training_label1)
        acc=clf.score(test_set1, test_label1)
        if acc > Accuracy:
            Accuracy = acc
            param[i] = [c]
    timetaken=time.time()-start
    return Accuracy,param,timetaken

#Function for RBF kernel
def svc_rbf(Cs,gammas,test_label1,test_set1,training_label1,training_set1,dfs,weight=None):
    Accuracy = 0
    param = {i:[]}
    start=time.time()
    for c in Cs:
        for g in gammas:
            clf = svm.SVC(kernel=k, C=c, gamma=g,decision_function_shape = dfs ,class_weight=weight).fit(training_set1, training_label1)
            acc=clf.score(test_set1, test_label1)
            if acc > Accuracy:
                Accuracy = acc
                param[i] = [c,g]
    timetaken=time.time()-start
    return Accuracy,param,timetaken

#Function for Sigmoid kernel
def svc_sigmoid(Cs,gammas,coef,test_label1,test_set1,training_label1,training_set1,dfs, weight=None):
    Accuracy = 0
    param = {i:[]}
    start=time.time()
    for c in Cs:
        for g in gammas:
            for r in coef:
                clf = svm.SVC(kernel=k, C=c,gamma=g, coef0=r,decision_function_shape =dfs, class_weight = weight).fit(training_set1, training_label1)
                acc=clf.score(test_set1, test_label1)
                if acc > Accuracy:
                    Accuracy = acc
                    param[i] = [c,g,r]
    timetaken=time.time()-start
    return Accuracy,param,timetaken

#Function for poly kernel
def svc_poly(Cs,gammas,coef,dgr, test_label1,test_set1,training_label1,training_set1,dfs, weight=None):
    Accuracy = 0
    param = {i:[]}
    start=time.time()
    for c in Cs:
        for g in gammas:
            for r in coef:
                for d in dgr:
                    clf = svm.SVC(kernel=k, C=c, gamma=g,coef0=r, degree=d, decision_function_shape=dfs, class_weight=weight).fit(training_set1, training_label1)
                    acc=clf.score(test_set1, test_label1)
                    if acc > Accuracy:
                        Accuracy = acc
                        param[i] = [c,g,r,d]
    timetaken=time.time()-start
    return Accuracy,param,timetaken
                
            ################################################
            ############### ONE VS ONE ####################
            ################################################
            
Dict={'linear':[],'rbf':[],'sigmoid':[],'poly':[]}
size=(int)(train.shape[0]/5)
for i in range(0,5):
    test_set=train[i*size:(i+1)*size,:]
    test_label=trainlabel[i*size:(i+1)*size,]
    training_set=np.delete(train,slice(i*size,(i+1)*size),0)
    training_label=np.delete(trainlabel,slice(i*size,(i+1)*size,))
    training_set1,test_set1,training_label1, test_label1 = train_test_split(training_set, training_label, test_size=0.2)   
    Cs = [ 2**-5,2**-3,2**-1,1, 2, 2**2]
    gammas = [2**-7, 2**-3,2**-13, 2**-5, 2**-1, 2**2]
    kernels = ['rbf','linear','sigmoid','poly']
    dgr=[2,3,1]
    coef=[0,0.1,0.01,0.001]
    decision_function_shape1 ='ovo'
    
    for k in kernels:
        if k =='linear':  
            Accuracy1,param1,timeTkenTOfindoptimalparam = svc_linear(Cs,test_label1,test_set1,training_label1,training_set1,decision_function_shape1)
#            print (" ")
#            print("For linear kernel : Max accuracy is ", Accuracy1," and hyperparameter C for ",i+1," fold is ",param1[i][0])
#            print (" ")
            Accuracy2,param2 , timetakenforfinal= svc_linear(param1[i],test_label,test_set,training_label,training_set,decision_function_shape1)
            print('')
            print ("The total time to execute is ",timeTkenTOfindoptimalparam+timetakenforfinal ," seconds" )
            print("For linear kernel : Accuracy for ",i+1," fold is ",Accuracy2*100," using hyperparameter C as ", param2[i][0])
            
            Dict[k].append(Accuracy2)
           
        elif k =='rbf':  
            Accuracy1,param1,timeTkenTOfindoptimalparam = svc_rbf(Cs,gammas,test_label1,test_set1,training_label1,training_set1,decision_function_shape1)
#            print (" ")
#            print("For rbf kernel :Max accuracy is ", Accuracy1," and hyperparameter C for ",i+1," fold is ",param1[i][0])
#            print (" ")
            a1=[param1[i][0]]
            a2=[param1[i][1]]
            Accuracy2,param2, timetakenforfinal = svc_rbf(a1,a2,test_label,test_set,training_label,training_set,decision_function_shape1)
            print (" ")
            print ("The total time to execute is ",timeTkenTOfindoptimalparam+timetakenforfinal ," seconds" )
            print("For rbf kernel : Accuracy for ",i+1," fold is ",Accuracy2*100," using hyperparameter C as ", param2[i][0]," gamma as ",param2[i][1])
            
            Dict[k].append(Accuracy2)
            
        elif k =='sigmoid':
            Accuracy1,param1,timeTkenTOfindoptimalparam = svc_sigmoid(Cs,gammas,coef,test_label1,test_set1,training_label1,training_set1,decision_function_shape1)
#            print (" ")
#            print("For sigmoid kernel : Max accuracy is ", Accuracy1," and hyperparameter C for ",i+1," fold is ",param1[i][0])
#            print (" ")
            a1=[param1[i][0]]
            a2=[param1[i][1]]
            a3=[param1[i][2]]
            Accuracy2,param2, timetakenforfinal = svc_sigmoid(a1,a2,a3,test_label,test_set,training_label,training_set,decision_function_shape1)
            print (" ")
            print ("The total time to execute is ",timeTkenTOfindoptimalparam+timetakenforfinal ," seconds" )       
            print("For sigmoid kernel : Accuracy for ",i+1," fold is ",Accuracy2*100," using hyperparameter C as ", param2[i][0]," gamma as ",param2[i][1], "coef as",param2[i][2])
            
            Dict[k].append(Accuracy2)
            
        else:
            Accuracy1,param1,timeTkenTOfindoptimalparam = svc_poly(Cs,gammas,coef,dgr,test_label1,test_set1,training_label1,training_set1,decision_function_shape1)
#            print (" ")
#            print("For poly kernel : Max accuracy is ", Accuracy1," and hyperparameter C for ",i+1," fold is ",param1[i][0])
#            print (" ")
            a1=[param1[i][0]]
            a2=[param1[i][1]]
            a3=[param1[i][2]]
            a4=[param1[i][3]]
            Accuracy2,param2, timetakenforfinal = svc_poly(a1,a2,a3,a4,test_label,test_set,training_label,training_set,decision_function_shape1)
            print (" ")
            print ("The total time to execute is ",timeTkenTOfindoptimalparam+timetakenforfinal ," seconds" )
            print("For poly kernel : Accuracy for ",i+1," fold is ",Accuracy2*100," using hyperparameter C as ", param2[i][0]," gamma as ",param2[i][1],"coef as",param2[i][2],"degree as",param2[i][3])
          
            Dict[k].append(Accuracy2)
            
print (" ")  
print("**Final Accuracy for linear kernel (ovo) is** ",np.sum(Dict['linear'])*100/5)
print (" ")
print("**Final Accuracy for rbf kernel (ovo) is** ",np.sum(Dict['rbf'])*100/5)
print (" ")
print("**Final Accuracy for sigmoid kernel (ovo) is** ",np.sum(Dict['sigmoid'])*100/5)
print (" ")
print("**Final Accuracy for poly kernel (ovo) is **",np.sum(Dict['poly'])*100/5)
#print (" ")

            ################################################
            ############### ONE VS REST ####################
            ################################################

Dict={'linear':[],'rbf':[],'sigmoid':[],'poly':[]}
size=(int)(train.shape[0]/5)
for i in range(0,5):
    test_set=train[i*size:(i+1)*size,:]
    test_label=trainlabel[i*size:(i+1)*size,]
    training_set=np.delete(train,slice(i*size,(i+1)*size),0)
    training_label=np.delete(trainlabel,slice(i*size,(i+1)*size,))
    training_set1,test_set1,training_label1, test_label1 = train_test_split(training_set, training_label, test_size=0.2)   
    Cs = [ 2**-5,2**-3,2**-1,1, 2, 2**2]
    gammas = [2**-7, 2**-3,2**-13, 2**-5, 2**-1, 2**2]
    dgr=[1,2,3]
    coef=[0,0.1,0.01,0.001]
    kernels = ['rbf','linear','sigmoid','poly']
    decision_function_shape1 ='ovr'
    
    
    for k in kernels:
        if k =='linear':  
            Accuracy1,param1,timeTkenTOfindoptimalparam = svc_linear(Cs,test_label1,test_set1,training_label1,training_set1,decision_function_shape1)
#            print (" ")
#            print("For linear kernel : Max accuracy is ", Accuracy1," and hyperparameter C for ",i+1," fold is ",param1[i][0])
#            print (" ")
            Accuracy2,param2 , timetakenforfinal= svc_linear(param1[i],test_label,test_set,training_label,training_set,decision_function_shape1)
            print('')
            print ("The total time to execute is ",timeTkenTOfindoptimalparam+timetakenforfinal ," seconds" )
            print("For linear kernel : Accuracy for ",i+1," fold is ",Accuracy2*100," using hyperparameter C as ", param2[i][0])
            
            Dict[k].append(Accuracy2)
           
        elif k =='rbf':  
            Accuracy1,param1,timeTkenTOfindoptimalparam = svc_rbf(Cs,gammas,test_label1,test_set1,training_label1,training_set1,decision_function_shape1)
#            print (" ")
#            print("For rbf kernel :Max accuracy is ", Accuracy1," and hyperparameter C for ",i+1," fold is ",param1[i][0])
#            print (" ")
            a1=[param1[i][0]]
            a2=[param1[i][1]]
            Accuracy2,param2, timetakenforfinal = svc_rbf(a1,a2,test_label,test_set,training_label,training_set,decision_function_shape1)
            print (" ")
            print ("The total time to execute is ",timeTkenTOfindoptimalparam+timetakenforfinal ," seconds" )
            print("For rbf kernel : Accuracy for ",i+1," fold is ",Accuracy2*100," using hyperparameter C as ", param2[i][0]," gamma as ",param2[i][1])
            
            Dict[k].append(Accuracy2)
            
        elif k =='sigmoid':
            Accuracy1,param1,timeTkenTOfindoptimalparam = svc_sigmoid(Cs,gammas,coef,test_label1,test_set1,training_label1,training_set1,decision_function_shape1)
#            print (" ")
#            print("For sigmoid kernel : Max accuracy is ", Accuracy1," and hyperparameter C for ",i+1," fold is ",param1[i][0])
#            print (" ")
            a1=[param1[i][0]]
            a2=[param1[i][1]]
            a3=[param1[i][2]]
            Accuracy2,param2, timetakenforfinal = svc_sigmoid(a1,a2,a3,test_label,test_set,training_label,training_set,decision_function_shape1)
            print (" ")
            print ("The total time to execute is ",timeTkenTOfindoptimalparam+timetakenforfinal ," seconds" )       
            print("For sigmoid kernel : Accuracy for ",i+1," fold is ",Accuracy2*100," using hyperparameter C as ", param2[i][0]," gamma as ",param2[i][1], "coef as",param2[i][2])
            
            Dict[k].append(Accuracy2)
            
        else:
            Accuracy1,param1,timeTkenTOfindoptimalparam = svc_poly(Cs,gammas,coef,dgr,test_label1,test_set1,training_label1,training_set1,decision_function_shape1)
#            print (" ")
#            print("For poly kernel : Max accuracy is ", Accuracy1," and hyperparameter C for ",i+1," fold is ",param1[i][0])
#            print (" ")
            a1=[param1[i][0]]
            a2=[param1[i][1]]
            a3=[param1[i][2]]
            a4=[param1[i][3]]
            Accuracy2,param2, timetakenforfinal = svc_poly(a1,a2,a3,a4,test_label,test_set,training_label,training_set,decision_function_shape1)
            print (" ")
            print ("The total time to execute is ",timeTkenTOfindoptimalparam+timetakenforfinal ," seconds" )
            print("For poly kernel : Accuracy for ",i+1," fold is ",Accuracy2*100," using hyperparameter C as ", param2[i][0]," gamma as ",param2[i][1],"coef as",param2[i][2],"degree as",param2[i][3])
          
            Dict[k].append(Accuracy2)
                        
print (" ")  
print("**Final Accuracy for linear kernel(ovr) is** ",np.sum(Dict['linear'])*100/5)       
print (" ")
print("**Final Accuracy for rbf kernel(ovr) is**",np.sum(Dict['rbf'])*100/5)
print (" ")
print("Final Accuracy for sigmoid kernel (ovr) is ",np.sum(Dict['sigmoid'])*100/5)
print (" ")
print("**Final Accuracy for poly kernel (ovr) is** ",np.sum(Dict['poly'])*100/5)
#print (" ")
            ################################################################
            ############### ONE VS ONE WITH CLASS WEIGHT ###################
            ################################################################
            
Dict={'linear':[],'rbf':[],'sigmoid':[],'poly':[]}
size=(int)(train.shape[0]/5)
for i in range(0,5):
    test_set=train[i*size:(i+1)*size,:]
    test_label=trainlabel[i*size:(i+1)*size,]
    training_set=np.delete(train,slice(i*size,(i+1)*size),0)
    training_label=np.delete(trainlabel,slice(i*size,(i+1)*size,))
    training_set1,test_set1,training_label1, test_label1 = train_test_split(training_set, training_label, test_size=0.2)   
    Cs = [ 2**-5,2**-3,2**-1,1, 2, 2**2]
    gammas = [2**-7, 2**-3,2**-13, 2**-5, 2**-1, 2**2]
    kernels = ['rbf','linear','sigmoid','poly']
    dgr=[1,2,3]
    coef=[0,0.1,0.01,0.001]
    decision_function_shape1 ='ovo'
    weight = 'balanced'
    
    for k in kernels:
        if k =='poly':  
            Accuracy1,param1,timeTkenTOfindoptimalparam = svc_linear(Cs,test_label1,test_set1,training_label1,training_set1,decision_function_shape1)
#            print (" ")
#            print("For linear kernel : Max accuracy is ", Accuracy1," and hyperparameter C for ",i+1," fold is ",param1[i][0])
#            print (" ")
            Accuracy2,param2 , timetakenforfinal= svc_linear(param1[i],test_label,test_set,training_label,training_set,decision_function_shape1)
            print('')
            print ("The total time to execute is ",timeTkenTOfindoptimalparam+timetakenforfinal ," seconds" )
            print("For linear kernel : Accuracy for ",i+1," fold is ",Accuracy2*100," using hyperparameter C as ", param2[i][0])
            
            Dict[k].append(Accuracy2)
           
        elif k =='rbf':  
            Accuracy1,param1,timeTkenTOfindoptimalparam = svc_rbf(Cs,gammas,test_label1,test_set1,training_label1,training_set1,decision_function_shape1,weight)
#            print (" ")
#            print("For rbf kernel :Max accuracy is ", Accuracy1," and hyperparameter C for ",i+1," fold is ",param1[i][0])
#            print (" ")
            a1=[param1[i][0]]
            a2=[param1[i][1]]
            Accuracy2,param2, timetakenforfinal = svc_rbf(a1,a2,test_label,test_set,training_label,training_set,decision_function_shape1,weight)
            print (" ")
            print ("The total time to execute is ",timeTkenTOfindoptimalparam+timetakenforfinal ," seconds" )
            print("For rbf kernel : Accuracy for ",i+1," fold is ",Accuracy2*100," using hyperparameter C as ", param2[i][0]," gamma as ",param2[i][1])
            
            Dict[k].append(Accuracy2)
            
        elif k =='sigmoid':
            Accuracy1,param1,timeTkenTOfindoptimalparam = svc_sigmoid(Cs,gammas,coef,test_label1,test_set1,training_label1,training_set1,decision_function_shape1, weight)
#            print (" ")
#            print("For sigmoid kernel : Max accuracy is ", Accuracy1," and hyperparameter C for ",i+1," fold is ",param1[i][0])
#            print (" ")
            a1=[param1[i][0]]
            a2=[param1[i][1]]
            a3=[param1[i][2]]
            Accuracy2,param2, timetakenforfinal = svc_sigmoid(a1,a2,a3,test_label,test_set,training_label,training_set,decision_function_shape1,weight)
            print (" ")
            print ("The total time to execute is ",timeTkenTOfindoptimalparam+timetakenforfinal ," seconds" )       
            print("For sigmoid kernel : Accuracy for ",i+1," fold is ",Accuracy2*100," using hyperparameter C as ", param2[i][0]," gamma as ",param2[i][1], "coef as",param2[i][2])          
            Dict[k].append(Accuracy2)
            
        else:
            Accuracy1,param1,timeTkenTOfindoptimalparam = svc_poly(Cs,gammas,coef,dgr,test_label1,test_set1,training_label1,training_set1,decision_function_shape1, weight)
#            print (" ")
#            print("For poly kernel : Max accuracy is ", Accuracy1," and hyperparameter C for ",i+1," fold is ",param1[i][0])
#            print (" ")
            a1=[param1[i][0]]
            a2=[param1[i][1]]
            a3=[param1[i][2]]
            a4=[param1[i][3]]
            Accuracy2,param2, timetakenforfinal = svc_poly(a1,a2,a3,a4,test_label,test_set,training_label,training_set,decision_function_shape1,weight)
            print (" ")
            print ("The total time to execute is ",timeTkenTOfindoptimalparam+timetakenforfinal ," seconds" )
            print("For poly kernel : Accuracy for ",i+1," fold is ",Accuracy2*100," using hyperparameter C as ", param2[i][0]," gamma as ",param2[i][1],"coef as",param2[i][2],"degree as",param2[i][3])
          
            Dict[k].append(Accuracy2)
            

print (" ")  
print("**Final Accuracy for linear kernel after applying balanced class weight is**", np.sum(Dict['linear'])*100/5)
print (" ")
print("**Final Accuracy for rbf kernel after applying balanced class weight is**",np.sum(Dict['rbf'])*100/5)
print (" ")
print("**Final Accuracy for sigmoid kernel after applying balanced class weight is** ",np.sum(Dict['sigmoid'])*100/5)
print (" ")
print("**Final Accuracy for poly kernel after applying balanced class weight is** ",np.sum(Dict['poly'])*100/5)
print (" ")