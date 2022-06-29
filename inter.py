import joblib
from sklearn.metrics import accuracy_score

# test_data和test_label为测试的数据和标签，格式为numpy
def svm_predict(test_data, test_label):
    # 导入SVM模型
    testmodel = joblib.load('svm.model')
    # 输出预测数组结果
    prediction = testmodel.predict(test_data)
    # 打印模型正确率
    print("accuracy: ", accuracy_score(prediction, test_label))
    # 返回预测数组
    return prediction