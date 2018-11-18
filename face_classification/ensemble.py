import pickle
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np


class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.estimator = weak_classifier
        self.estimators = []
        self.n_estimators = n_weakers_limit
        self.alpha = 1
        self.alphas = []
        self.sample_weight = None
        self.learning_rate = 1
        pass

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self, X, y):
        '''Build a boosted classifier from the training set (X, y).

        Returns:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        #初始化样本权重
        self.sample_weight = np.empty(X.shape[0], dtype=np.float64)
        self.sample_weight[:] = 1. / X.shape[0]
        for iboost in range(self.n_estimators):
            # 使用有权值分布w的训练数据集学习得到弱分类器
            self.estimator.fit(X, y, sample_weight=self.sample_weight.reshape(-1))
            # 计算分类误差率
            y_predict = self.estimator.predict(X).reshape(-1, 1)
            incorrect = y_predict != y
            estimator_error = np.mean(np.average(incorrect, weights=self.sample_weight, axis=0))
            
            if estimator_error <= 0:
                self.alphas.append(self.alpha)
                self.estimators.append(self.estimator)
                continue
            else:
                # 计算弱分类器在强分类器中的比重alpha
                self.alpha = 0.5 * np.log((1-estimator_error) / estimator_error)
                # 更新样本权值分布
                self.sample_weight = self.sample_weight.reshape(
                    -1, 1)*np.exp(-self.alpha*y.reshape(-1, 1)*y_predict.reshape(-1, 1))
                self.alphas.append(self.alpha)
                self.estimators.append(self.estimator)
        return X, y

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        scores = np.empty(X.shape[0], dtype=np.float64).reshape(-1, 1)
        for iboost in range(self.n_estimators):
            scores = np.concatenate(
                (1.*self.alphas[iboost] *
                 self.estimators[iboost].predict_proba(X)[:, 1].reshape(-1, 1)
                 / (self.estimators[iboost].predict_proba(X)[:, 1].reshape(-1, 1) +
                    self.estimators[iboost].predict_proba(X)[:, 0].reshape(-1, 1)), scores), axis=1)
        return np.average(scores[:, 0:-1], axis=1).reshape(-1, 1)

    def predict(self, X, threshold=0.5):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        score = self.predict_scores(X)
        df = pd.DataFrame(score)
        df[1] = df[0].apply(lambda x: 1 if x > threshold else -1)
        return np.array(df[1]).reshape(-1, 1)

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
