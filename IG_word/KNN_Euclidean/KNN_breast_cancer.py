# -*- coding:utf-8 -*-
import csv
import random
import math
import operator
import pandas as pd
def loadDataset(filename,split,trainingSet=[],testSet=[]):
    with open(filename, "rt", encoding="utf-8") as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines)
        for x in range(len(dataset)-1):
            for y in range(9):
                dataset[x][y]=float(dataset[x][y])
            if random.random()<split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

def euclideanDistance(instance1,instance2,length):
    distance=0
    for x in range(length):
        distance+=pow((instance1[x]-instance2[x]),2)
    return math.sqrt(distance)

def getNeighbors(trainingSet,testInstance,k):
    distances=[]
    length=len(testInstance)-1
    for x in range(len(trainingSet)):
        dist=euclideanDistance(testInstance,trainingSet[x],length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response=neighbors[x][-1]
        if response in classVotes:
            classVotes[response]+=1
        else:
            classVotes[response]=1
    sortedVotes=sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]
def getAccurcy(testSet,predictions):
    correct=0
    for x in range(len(testSet)-1):
        if testSet[x][-1]==predictions[x]:
            correct+=1
    return (correct/float(len(testSet)))*100.0

def main():
    trainingSet=[]
    testSet=[]
    split=0.67
    loadDataset(r'E:\资料\数据集dataset\breast_cancer.csv',split,trainingSet,testSet)
    print ('train set='+repr(trainingSet))
    print ('test set='+repr(testSet))
    predictions=[]
    k=3
    for x in range(len(testSet)):
        neighbors=getNeighbors(trainingSet,testSet[x],k)
        result=getResponse(neighbors)
        predictions.append(result)
        print('predicted='+repr(result)+',actual='+repr(testSet[x][-1]))
    accuracy=getAccurcy(testSet,predictions)
    print('Accuracy:'+repr(accuracy)+'%')
main()