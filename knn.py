import time
import joblib
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import os
import struct
import numpy as np
import pickle

'''
函数说明：从文件中读取mnist数据集
'''
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

'''
函数说明：将手写数字图片画出来
'''
def plotImg(imgArrays):
    fig, ax = plt.subplots(
        nrows=int(len(imgArrays)/5) + 1,
        ncols=5,
        sharex=True,
        sharey=True, )
    ax = ax.flatten()
    for i in range(len(imgArrays)):
        img = imgArrays[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
def loadResult():
    with open("result.txt", 'rb') as f:
        result = pickle.load(file=f)
    return result
'''
函数说明：构建基分类器，并获得模型性能参数和分类结果
'''
def classfier(classfierName):
    imgs, labels = load_mnist(IMG_PATH, kind="train")
    testImgs, testLabel = load_mnist(IMG_PATH, kind="test")
    # 构建kNN分类器
    print(f"{classfierName}分类器：")
    startTime = time.time()
    if classfierName == "knn":
        Aclassfier = KNN(n_neighbors=3, algorithm='auto')
    elif classfierName == "ann":
        Aclassfier = MLPClassifier()
    # 拟合模型, trainingMat为训练矩阵,hwLabels为对应的标签
    Aclassfier.fit(np.array(imgs[0:3000]), np.array(labels[0:3000]))
    print(f"{classfierName}模型训练时间：{(time.time() - startTime)}")
    # joblib.dump(Aclassfier, f'{classfierName}.model')
    # knnClassfier = joblib.load('knn.model')
    score = Aclassfier.score(np.array(testImgs[0:10000]), np.array(testLabel[0:10000]))
    print(f"{classfierName}模型训练时间+测试时间：{(time.time() - startTime)}")
    print(f"{classfierName}分类器正确率：{score}")
    print("------------------------------------------------------")
    joblib.dump(Aclassfier, f'{classfierName}.model')
'''
knn预测函数，预测结果并保存起来，将后40000组预测结果整出来
'''
def knn_test():
    img1, trainLabels = load_mnist(IMG_PATH, 'train')
    img2, testLabels = load_mnist(IMG_PATH, 'test')
    testImgs = np.append(np.array(img1[30000:60000]), np.array(img2[0:10000]), axis=0)
    knnClassfier = joblib.load('knn.model')
    knnResult = knnClassfier.predict(testImgs)
    with open("result.txt", 'wb') as f:
        pickle.dump(knnResult, f)

'''
预测函数，使用knn模型，直接调用已经算好的结果
输入测试集范围，返回预测结果数组
'''
def knn_predict1(start, end):
    img, trainLabels = load_mnist(IMG_PATH, 'train')
    img, testLabels = load_mnist(IMG_PATH, 'test')
    myLabels = np.append(trainLabels[30000:60000], testLabels)
    knnResult = loadResult()
    accuracy = accuracy_score(myLabels[start:end], knnResult[start:end])
    print(f"accuracy: {accuracy}")
    return knnResult[start:end]

def ann_predict(testImgs, testLabels):
    annClassfier = joblib.load('ann.model')
    annResult = annClassfier.predict(testImgs)
    accuracy = accuracy_score(testLabels, annResult)
    print(f"accuracy: {accuracy}")
    return annResult

def knn_predict(testImgs, testLabels):
    knnClassfier = joblib.load('knn.model')
    knnResult = knnClassfier.predict(testImgs)
    accuracy = accuracy_score(testLabels, knnResult)
    print(f"accuracy: {accuracy}")
    return knnResult

IMG_PATH = os.getcwd() + "\\img\\"
"""
函数说明:main函数
KNN测试结果建议直接读取序列化结果（'result.txt'）,60000的测试集检测一个测试数据大概需要0.1s
"""
# classfier('knn')
# classfier('ann')
# classfier("knn")
# knn_test()

# classfier("ann")

# print(p1)
# p1 = loadResult()[60000:60100]
# p2 = classfier("ANN")
# # 直接SVM不太行
# # p3 = classfier("SVM")
# print(p1, p2)

# imgs, labels = load_mnist(IMG_PATH, kind='test')
# # # print(labels[:20])
# re1 = cnn_predict(imgs[:2000], labels[:2000])
# print("cnn:", re1)
# re2 = svm_predict(imgs[:20], labels[:20])
# print("svm:", re2)
# re3 = knn_predict(imgs, labels)
# print("knn:", re3)
# re4 = ann_predict(imgs[:20], labels[:20])
# print("ann:", re4)



