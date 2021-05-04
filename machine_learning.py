from sklearn import svm
from sklearn import tree
from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np


def svm_predict(x_train, train_labels, x_test):
    clf = svm.SVC(kernel='rbf', gamma=0.5)
    clf.fit(x_train, train_labels)
    predicted = clf.predict(x_test)
    return predicted, clf


def dt_predict(x_train, train_labels, x_test):
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, train_labels)
    predicted = clf.predict(x_test)
    return predicted, clf


def record_stats(data_labels, predicts, pos_label=None, average=None):
    stats = [0, 0, 0]
    if pos_label is None:
        stats[0] = f1_score(data_labels, predicts, average=average)
        stats[1] = precision_score(data_labels, predicts, average=average)
        stats[2] = recall_score(data_labels, predicts, average=average)
    else:
        stats[0] = f1_score(data_labels, predicts, pos_label=pos_label)
        stats[1] = precision_score(data_labels, predicts, pos_label=pos_label)
        stats[2] = recall_score(data_labels, predicts, pos_label=pos_label)
    return stats


def remove_n_features(num, train_data, train_labels, test_data, test_labels):

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    index = 0
    removed = []

    for n in range(num):
        max_score = [0,0,0]
        for i in range(train_data.shape[1]):
            # remove feature i from both train and test
            temp_train = np.delete(train_data, i, 1)
            temp_test = np.delete(test_data, i, 1)

            # create classifier
            clf = svm.SVC(kernel='rbf', gamma=0.5)
            clf.fit(temp_train, train_labels)

            # predict
            predicted = clf.predict(temp_test)

            stats = record_stats(test_labels, predicted, average='weighted')

            if stats[0] > max_score[0]:
                index = i
                max_score = stats
                print('best stats so far: {}, {}'.format(stats[0], i))
        train_data = np.delete(train_data, index, 1)
        test_data = np.delete(test_data, index, 1)
        temp = removed.copy()
        temp.sort()
        for k in range(len(removed)):
            if temp[k] <= index:
                index += 1
        removed.append(index)
    print('Removed features: {}'.format(removed))
    return removed, train_data, test_data, max_score


def add_n_features(num, train_data, train_labels, test_data, test_labels):
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    index = 0
    added = []
    temp_train = []
    temp_test = []

    for n in range(num):
        max_score = [0,0,0]
        for i in range(train_data.shape[1]):
            if i not in added:
                # add feature i from both train and test
                if n == 0:
                    temp_train = train_data[:, i].reshape(-1,1)
                    temp_test = test_data[:, i].reshape(-1,1)
                else:
                    temp_train = np.column_stack((temp_train, train_data[:, i]))
                    temp_test = np.column_stack((temp_test, test_data[:, i]))

                # create classifier
                clf = svm.SVC(kernel='rbf', gamma=0.5)
                clf.fit(temp_train, train_labels)

                # predict
                predicted = clf.predict(temp_test)

                stats = record_stats(test_labels, predicted, average='weighted')

                if stats[0] > max_score[0]:
                    index = i
                    max_score = stats
                    print('best stats so far: {}, {}'.format(stats[0], i))
        return_train = np.append(train_data, temp_train, 1)
        return_test = np.append(test_data, temp_test, 1)
        added.append(index)
    print('Added features: {}'.format(added))
    return added, return_train, return_test, max_score
