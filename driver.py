import machine_learning as ml
import plot
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.inspection import permutation_importance
import numpy as np


def experiment(data, outfile, binary):
    temp = 'Ternary'
    if binary:
        temp = 'Binary'

    svm_pred, svm = ml.svm_predict(data['train'], data['train_labels'],
                                   data['test'])

    if binary:
        svm_stats = ml.record_stats(data['test_labels'], svm_pred, pos_label='QSO')
    else:
        svm_stats = ml.record_stats(data['test_labels'], svm_pred, average='weighted')

    with open(outfile, 'w') as f:
        f.write('{} SVM:'.format(temp))
        f.write('F1: {}\nPrecision: {}\nRecall: {}\n\n'.format(svm_stats[0], svm_stats[1], svm_stats[2]))

    dt_pred, dt = ml.dt_predict(data['train'], data['train_labels'],
                                        data['test'])

    if binary:
        dt_stats = ml.record_stats(data['test_labels'], dt_pred, pos_label='QSO')
    else:
        dt_stats = ml.record_stats(data['test_labels'], dt_pred, average='weighted')

    importances = permutation_importance(svm, data['train'], data['train_labels'], n_repeats=10, random_state=0)

    plot.bar_graph(importances['importances_mean'], importances['importances_std'])

    with open(outfile, 'a') as f:
        f.write('{} Decision Tree:'.format(temp))
        f.write('F1: {}\nPrecision: {}\nRecall: {}\n\n'.format(dt_stats[0], dt_stats[1], dt_stats[2]))
        f.write('Feature importances: {}\n'.format(dt.feature_importances_))
        f.write('Permutation Feature Importances: {}'.format(importances))

    removed_features, trimmed_train, \
    trimmed_test, remove_f1 = ml.remove_n_features(len(data['train'][0]) - 2, data['train'],data['train_labels'],
                                                      data['test'],data['test_labels'])

    added_features, trimmed_train, \
    trimmed_test, add_f1 = ml.add_n_features(2, data['train'], data['train_labels'],
                                                 data['test'], data['test_labels'])

    with open(outfile, 'a') as f:
        f.write('Sequential Backward Selection Results:\n')
        f.write('Score (F1, Precision, Recall): {}\n'.format(remove_f1))
        f.write('Removed features: {}\n\n'.format(removed_features))
        f.write('Sequential Forward Selection Results:\n')
        f.write('Score (F1, Precision, Recall): {}\n'.format(add_f1))
        f.write('Added features: {}\n\n'.format(added_features))

    if add_f1[0] > remove_f1[0]:
        twod_data = {'train': np.array(data['train']), 'test': np.array(data['test'])}
        twod_data['train'] = twod_data['train'][:, [added_features[0], added_features[1]]].reshape(-1, 2)
        twod_data['test'] = twod_data['test'][:, [added_features[0], added_features[1]]].reshape(-1, 2)
    else:
        twod_data = {'train': np.array(data['train']), 'test': np.array(data['test'])}
        twod_data['train'] = twod_data['train'][:, [removed_features[0], removed_features[1]]].reshape(-1, 2)
        twod_data['test'] = twod_data['test'][:, [removed_features[0], removed_features[1]]].reshape(-1, 2)

    pca = PCA(n_components=len(data['train'][0])-1)
    pca.fit(data['train'])

    pca_train = pca.transform(data['train'])
    pca_test = pca.transform(data['test'])

    svm_pred, svm = ml.svm_predict(pca_train, data['train_labels'],
                                   pca_test)

    if binary:
        svm_stats = ml.record_stats(data['test_labels'], svm_pred, pos_label='QSO')
    else:
        svm_stats = ml.record_stats(data['test_labels'], svm_pred, average='weighted')

    with open(outfile, 'a') as f:
        f.write('{} SVM with PCA:'.format(temp))
        f.write('F1: {}\nPrecision: {}\nRecall: {}\n\n'.format(svm_stats[0], svm_stats[1], svm_stats[2]))

    lda = LDA()
    lda.fit(data['train'], data['train_labels'])

    lda_train = lda.transform(data['train'])
    lda_test = lda.transform(data['test'])

    lda_pred = lda.predict(data['test'])

    svm_pred, svm = ml.svm_predict(lda_train, data['train_labels'],
                                   lda_test)
    if binary:
        svm_stats = ml.record_stats(data['test_labels'], svm_pred, pos_label='QSO')
    else:
        svm_stats = ml.record_stats(data['test_labels'], svm_pred, average='weighted')

    with open(outfile, 'a') as f:
        f.write('{} SVM with LDA:'.format(temp))
        f.write('F1: {}\nPrecision: {}\nRecall: {}\n\n'.format(svm_stats[0], svm_stats[1], svm_stats[2]))

    lda = LDA()
    lda.fit(twod_data['train'], data['train_labels'])
    lda_train = lda.transform(twod_data['train'])
    lda_test = lda.transform(twod_data['test'])
    svm_pred, svm = ml.svm_predict(lda_train, data['train_labels'],
                                   lda_test)

    if binary:
        svm_stats = ml.record_stats(data['test_labels'], svm_pred, pos_label='QSO')
    else:
        svm_stats = ml.record_stats(data['test_labels'], svm_pred, average='weighted')

    with open(outfile, 'a') as f:
        f.write('{} 2D SVM with LDA:'.format(temp))
        f.write('F1: {}\nPrecision: {}\nRecall: {}\n\n'.format(svm_stats[0], svm_stats[1], svm_stats[2]))

    plot.svm_graph(twod_data['train'], data['train_labels'], binary)
    plot.svm_graph(twod_data['test'], data['test_labels'], binary)
    #plot.vis_tree(dt)