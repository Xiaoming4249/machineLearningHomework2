from sklearn import svm
import numpy as np
from time import time
from sklearn.metrics import accuracy_score
from struct import unpack
from sklearn.model_selection import GridSearchCV
import joblib
import os

def readimage(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img


def readlabel(path):
    with open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.fromfile(f, dtype=np.uint8)
    return lab


def main():
    train_data = readimage(IMG_PATH+"train-images.idx3-ubyte")
    train_label = readlabel(IMG_PATH+"train-labels.idx1-ubyte")
    test_data = readimage(IMG_PATH+"test-images.idx3-ubyte")
    test_label = readlabel(IMG_PATH+"test-labels.idx1-ubyte")
    svc = svm.SVC()
    parameters = {'kernel': ['rbf'], 'C': [1]}
    print("Train...")
    clf = GridSearchCV(svc, parameters, n_jobs=-1)
    start = time()
    clf.fit(train_data[0:30000], train_label[0:30000])
    joblib.dump(clf, 'svm.model')
    end = time()
    t = end - start
    print('Train：%dmin%.3fsec' % (t // 60, t - 60 * (t // 60)))
    # testmodel = joblib.load('svm.model')
    # prediction = testmodel.predict(test_data)
    # print("accuracy: ", accuracy_score(prediction, test_label))
    # accurate = [0] * 10
    # sumall = [0] * 10
    # i = 0
    # while i < len(test_label):
    #     sumall[test_label[i]] += 1
    #     if prediction[i] == test_label[i]:
    #         accurate[test_label[i]] += 1
    #     i += 1
    # print("分类正确的：", accurate)
    # print("总的测试标签：", sumall)

IMG_PATH = os.getcwd() + "\\img\\"
if __name__ == '__main__':
    main()