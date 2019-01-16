# -*- coding:utf-8 -*-
from IG_word.TF_IDF import *
import numba as nb

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def cosineDistance(instance1, instance2):
    dist1 = 1 - np.dot(instance1, instance2)/(np.linalg.norm(instance1)*np.linalg.norm(instance2))
    return dist1
@nb.jit
def space_density_distance(trainingSet,instance1,instance2,length):
    distance = 0
    max_list = []
    min_list = []
    for j in range(length):
        one_list = []
        for i in range(len(trainingSet)):
            one_list.append(float(trainingSet[i][j]))
        max_list.append(max(one_list))
        min_list.append(min(one_list))
    u = 1
    v = 1
    for x in range(length):
        test_distance = 0
        train_distance = 0
        for z in range(len(trainingSet)):
            test_distance += math.pow((trainingSet[z][x]-instance1[x]), 2) / (trainingSet[z][x]+instance1[x]+0.1)
        for y in range(len(trainingSet)):
            train_distance += math.pow((trainingSet[y][x]-instance2[x]), 2) / (trainingSet[y][x]+instance2[x]+0.1)
        distance += (math.pow(instance1[x]-instance2[x], 2)/((instance1[x]+instance2[x])+0.1))*(u*math.sqrt(test_distance/(test_distance + train_distance)) + v*math.sqrt(test_distance/ (train_distance)))
    return distance
# def space_density_distance(trainingSet,instance1,instance2,length,value):
#     distance = 0
#     for x in range(length):
#         test_distance = 0
#         train_distance = 0
#         for z in range(len(trainingSet)):
#             test_distance += abs(trainingSet[z][x]-instance1[x])/value
#         for y in range(len(trainingSet)):
#             train_distance += abs(trainingSet[y][x]-instance2[x])/value
#         distance += (abs(instance1[x]-instance2[x])/value)*(math.sqrt(test_distance/(test_distance + train_distance)) + math.sqrt(test_distance/ train_distance))
#     return distance
@nb.jit
def getNeighbors(trainingSet, testInstance, k, label):
    distances = []
    length = len(testInstance)
    # max_list = []
    # min_list = []
    # for j in range(length):
    #     one_list = []
    #     for i in range(len(trainingSet)):
    #         one_list.append(float(trainingSet[i][j]))
    #     max_list.append(max(one_list))
    #     min_list.append(min(one_list))
    # value = max(max_list) - min(min_list)
    for x in range(len(trainingSet)):
        dist = space_density_distance(trainingSet,testInstance, trainingSet[x], length)
        #dist = cosineDistance(testInstance, trainingSet[x])
        distances.append((label[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

@nb.jit
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

@nb.jit
def getAccurcy(test_label, predictions):
    correct = 0
    for x in range(len(test_label)):
        if test_label[x] == predictions[x]:
            correct += 1
    return (correct / len(test_label)) * 100.0

@nb.jit
def testbagOfWord2Vec(vocabList, inputSet):  # 词袋模型，统计概率的
    tmp = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            tmp[vocabList.index(word)] += 1  # 当前文档有这个词条，则根据词典位置获取其位置并赋值为1
    return tmp
@nb.jit
def euclidean(instance1, instance2):  # 计算测试文本和每个训练集文本的距离
    distance = 0
    length = len(instance2)
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)
@nb.jit
def density_compute(trainingSet):  # 训练集的距离矩阵
    dis = np.zeros((len(trainingSet), len(trainingSet)))  # 距离矩阵
    disvec = np.zeros((int((len(trainingSet) * (len(trainingSet) - 1)) / 2), 1))  # 所有距离向量
    k = 1
    for i in range(len(trainingSet)):
        for j in range(i-1):
            dis[i, j] = euclidean(trainingSet[i], trainingSet[j])     # 计算两两文本之间的距离
            dis[j, i] = dis[i, j]
            disvec[k] = dis[i, j]
            k = k+1

    disvec = np.sort(disvec)  # 升序排列
    dist_in = np.sum(disvec)/len(disvec)  # 求解平均距离
    percent = 0.15
    dc = disvec[round(percent * len(disvec))]  # 选取邻域参数dc,round函数默认保留为整数
    # 计算密度向量p
    p = np.zeros((len(trainingSet), 1))
    for j in range(len(trainingSet)):
        for q in range(len(trainingSet)):
            p[j] += np.exp(-math.pow(dis[j][q]/dc, 2))
        p[j] = p[j] - np.exp(-math.pow(dis[j][j]/dc, 2))  # 高斯核函数 除去自身
    return disvec, p, dc, dist_in
@nb.jit
def test_density_compute(trainingSet, testing, dc):
    dict_value = np.zeros(len(trainingSet))
    for i in range(len(trainingSet)):
            dict_value[i] = euclidean(trainingSet[i], testing)     # 计算两两文本之间的距离
    x = 0
    for j in range(len(trainingSet)):
            x += np.exp(-math.pow(dict_value[j]/dc, 2))
    x = x - np.exp(-math.pow(dict_value[j]/dc, 2))  # 高斯核函数 除去自身
    return x
@nb.jit
def test_density(train_vec_List, test_vec_List, dc):
    test_density = np.zeros(len(test_vec_List))
    for i in range(len(test_vec_List)):
        test_density[i] = test_density_compute(train_vec_List, test_vec_List[i], dc)
    return test_density
@nb.jit
def KNN_classify():
    train_vec_List, idf_array, label = tf_idf()
    test_vec_List, test_label = test_tf_idf()
    # disvec, p, dc, dist_in = density_compute(train_vec_List)  # 训练集的密度
    # test_density_value = test_density(train_vec_List, test_vec_List, dc)
    k = 25
    prediction = []
    for x in range(len(test_vec_List)):
        neighbors = getNeighbors(train_vec_List, test_vec_List[x], k, label)
        result = getResponse(neighbors)
        prediction.append(result)
        print("prediction=" + repr(result) + "  actual=" + repr(test_label[x]))
    accuracy = getAccurcy(test_label, prediction)
    return accuracy

if __name__ == '__main__':
    # 测试文本，使用KNN分类
    print(KNN_classify())
