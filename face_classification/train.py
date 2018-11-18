from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from PIL import Image
import feature
import ensemble
import numpy as np
import codecs

sample_num = 500
size = 24, 24
face_feature = []
nonface_feature = []
weak_clf = DecisionTreeClassifier(max_depth=5)
weak_clf_num = 6

# 提取图片特征函数
def extract_feature(pic_type, feature_type):
    path_name = './datasets/original/' + pic_type + \
        '/' + pic_type + '_' + '%03d' % i+'.jpg'
    obj = Image.open(path_name).convert('L')
    obj.thumbnail(size, Image.ANTIALIAS)
    npd = feature.NPDFeature(np.array(obj))
    n = npd.extract()
    feature_type.append(n.tolist())


# 提取特征
for i in range(sample_num):
    extract_feature(pic_type='face', feature_type=face_feature)
    extract_feature(pic_type='nonface', feature_type=nonface_feature)

# 将数据集切分为训练集和验证集
temp_pos = np.ones(sample_num)
temp_neg = -temp_pos
all_attribute = np.concatenate((face_feature, nonface_feature), axis=0)
all_label = np.concatenate(
    (temp_pos.reshape(-1, 1), temp_neg.reshape(-1, 1)), axis=0)
X_train, X_valid, Y_train, Y_valid = train_test_split(
    all_attribute, all_label, test_size=0.2)

# 建立强分类器
clf = ensemble.AdaBoostClassifier(
    weak_classifier=weak_clf, n_weakers_limit=weak_clf_num)
clf.fit(X_train, Y_train)

# 使用强分类器对验证集进行预测
y_pred = clf.predict(X_valid)
y_pred = y_pred.reshape(-1).tolist()
y_true = Y_valid.reshape(-1).tolist()

# 写入预测结果
fout = codecs.open('classifier_report.txt', 'w', 'utf-8')
target_names = ['face', 'nonface']
result = classification_report(y_true, y_pred, target_names=target_names)
fout.write(result)
fout.close()

print(result)

