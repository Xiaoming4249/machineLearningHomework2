from sklearn import svm
import numpy as np
from time import time
from sklearn.metrics import accuracy_score
from struct import unpack
from sklearn.model_selection import GridSearchCV
import joblib

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
    test_data = readimage("t10k-images.idx3-ubyte")
    test_label = readlabel("t10k-labels.idx1-ubyte")
    testmodel = joblib.load('svm.model')
    prediction = testmodel.predict(test_data)
    print("accuracy: ", accuracy_score(prediction, test_label))
    accurate = [0] * 10
    sumall = [0] * 10
    i = 0
    while i < len(test_label):
        sumall[test_label[i]] += 1
        if prediction[i] == test_label[i]:
            accurate[test_label[i]] += 1
        i += 1
    print("分类正确的：", accurate)
    print("总的测试标签：", sumall)


if __name__ == '__main__':
    main()