import time
from sklearn import svm
import numpy as np
from knn import knn_predict, load_mnist, IMG_PATH
from knn import ann_predict
from inter import svm_predict
from cnn import CNN, cnn_predict
from mydata import readPng
if __name__ == '__main__':

    '''
    元学习器，首先获取四个基学习器的模型测试结果，然后使用SVM模型训练元学习器，最后使用测试集数据比较五个分类器的正确率和预测时间
    '''
    # 获取样本
    trainImgs, trainLabels = load_mnist(IMG_PATH, 'train')
    testImgs, testLabels = load_mnist(IMG_PATH, 'test')
    print("前30000~60000样本各分类器测试效果：")
    # knn
    startTime = time.time()
    print("knn:")
    knn_re = knn_predict(trainImgs[30000:60000], trainLabels[30000:60000])
    print(f"测试时间：{(time.time() - startTime)}")
    # ann
    print("ann:")
    startTime = time.time()
    ann_re = ann_predict(trainImgs[30000:60000], trainLabels[30000:60000])
    print(f"测试时间：{(time.time() - startTime)}")
    # cnn
    print("cnn:")
    startTime = time.time()
    cnn_re = cnn_predict(trainImgs[30000:60000], trainLabels[30000:60000])
    print(f"测试时间：{(time.time() - startTime)}")
    # svm
    print("svm:")
    startTime = time.time()
    svm_re = svm_predict(trainImgs[30000:60000], trainLabels[30000:60000])
    print(f"测试时间：{(time.time() - startTime)}")

    # 根据基分类器分类结果，建立元分类器训练集
    yuan_train = []
    yuan_labels = np.array(trainLabels[30000:60000])
    for i in range(30000):
        row = [knn_re[i], ann_re[i], cnn_re[i], svm_re[i]]
        yuan_train.append(row)
    # 根据基分类器分类结果，建立元分类器训练集，不含svm
    # yuan_train = []
    # yuan_labels = np.array(trainLabels[30000:60000])
    # for i in range(30000):
    #     row = [knn_re[i], ann_re[i], cnn_re[i]]
    #     yuan_train.append(row)

    # 基于sklearn的svm分类器，构建元学习器
    startTime = time.time()
    svc = svm.SVC(kernel='rbf', C=1)
    svc.fit(np.array(yuan_train), yuan_labels)
    print(f"元学习器训练时间：{(time.time() - startTime)}")

    startTime0 = time.time()
    # 运行各基分类器，获得元学习器的测试集输入
    '''
    # Mnist测试集
    # knn
    print("knn:")
    startTime = time.time()
    knn_tre = knn_predict(testImgs, testLabels)
    print(f"测试时间：{(time.time() - startTime)}")
    # ann
    print("ann:")
    startTime = time.time()
    ann_tre = ann_predict(testImgs, testLabels)
    print(f"测试时间：{(time.time() - startTime)}")
    # cnn
    print("cnn:")
    startTime = time.time()
    cnn_tre = cnn_predict(testImgs, testLabels)
    print(f"测试时间：{(time.time() - startTime)}")
    # svm
    print("svm:")
    startTime = time.time()
    svm_tre = svm_predict(testImgs, testLabels)
    print(f"测试时间：{(time.time() - startTime)}")
    # 构建元测试集,含SVM
    yuan_test = []
    for i in range(10000):
        row = [knn_tre[i], ann_tre[i], cnn_tre[i], svm_tre[i]]
        yuan_test.append(row)
    # # 构建元测试集,不含SVM
    # yuan_test = []
    # for i in range(10000):
    #     row = [knn_tre[i], ann_tre[i], cnn_tre[i]]
    #     yuan_test.append(row)
    '''
    # 自建样本
    testImgs, testLabels = readPng()
    # knn
    print("knn:")
    startTime = time.time()
    knn_tre = knn_predict(testImgs, testLabels)
    print(f"测试时间：{(time.time() - startTime)}")
    # ann
    print("ann:")
    startTime = time.time()
    ann_tre = ann_predict(testImgs, testLabels)
    print(f"测试时间：{(time.time() - startTime)}")
    # cnn
    print("cnn:")
    startTime = time.time()
    cnn_tre = cnn_predict(testImgs, testLabels)
    print(f"测试时间：{(time.time() - startTime)}")
    # svm
    print("svm:")
    startTime = time.time()
    svm_tre = svm_predict(testImgs, testLabels)
    print(f"测试时间：{(time.time() - startTime)}")
    # 构建元测试集,含SVM
    yuan_test = []
    for i in range(len(testImgs)):
        row = [knn_tre[i], ann_tre[i], cnn_tre[i], svm_tre[i]]
        yuan_test.append(row)

    score = svc.score(yuan_test, testLabels)
    print(f"元学习器训练时间+测试时间：{(time.time() - startTime0)}")
    print(f"元学习器正确率：{score}")
    print("------------------------------------------------------")


