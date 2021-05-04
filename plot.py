from sklearn import tree
from matplotlib import pyplot as plt
import pydotplus
import numpy as np
from sklearn import svm

fn = ['u', 'g', 'r', 'i', 'z', 'w1', 'w2', 'w3','w4', 'psf_r', 'psf_z']
cn = ['QSO', 'GALAXY', 'STAR']

def vis_tree(clf):
    dot_data = tree.export_graphviz(clf, feature_names=fn,
                                    class_names=cn, filled=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('graph.png')

def make_meshgrid(x, y, h=0.2):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def convert_labels(train_labels, binary=False):
    if binary:
        for i in range(len(train_labels)):
            if train_labels[i] == 'QSO':
                train_labels[i] = 0
            if train_labels[i] == 'NOT QSO':
                train_labels[i] = 1
    else:
        for i in range(len(train_labels)):
            if train_labels[i] == 'QSO':
                train_labels[i] = 0
            if train_labels[i] == 'GALAXY':
                train_labels[i] = 1
            if train_labels[i] == 'STAR':
                train_labels[i] = 2
    return train_labels


def svm_graph(train, train_labels, binary=False):
    fig,ax = plt.subplots()
    title= 'Decision Boundary for 2D SVM'
    x0, x1 = train[:,0], train[:,1]
    xx,yy = make_meshgrid(x0,x1)
    clf = svm.SVC(kernel='rbf', gamma=0.5)
    train_labels = convert_labels(train_labels, binary)
    clf.fit(train, train_labels)
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    out = ax.contourf(xx,yy,z, cmap=plt.cm.rainbow, alpha=0.8)
    ax.scatter(x0,x1,c=train_labels, cmap=plt.cm.rainbow, s=20, edgecolors='k')
    ax.set_ylabel('Feature 1')
    ax.set_xlabel('Feature 2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.show()


def bar_graph(data, err_bars):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(data)), data, yerr=err_bars, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('F-score')
    ax.set_xticks(fn)
    ax.set_title('Feature Permutation Importances')
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.show()
