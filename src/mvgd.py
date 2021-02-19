import numpy as np

#precision, recall, F1
#precision = tp / tp + fp
#recall = tp / tp + fn
#F1
def gx(mu, sig, point):
    point = np.array([point])
    W = np.matmul((mu-point), sig.transpose())
    value = - 0.5 * np.matmul(W, (mu-point).transpose())
    value -= -0.5 * np.log(np.linalg.det(sig))
    return value


def classify(mu1, sig1, mu2, sig2, datac1, datac2):
    tp = fn = 0
    fp = tn = 0
    for point in datac1:
        if gx(mu1, sig1, point) < gx(mu2, sig2, point):
            fn += 1 #false negative
        else:
            tp += 1 #true positive
    for point in datac2:
        if gx(mu2, sig2, point) < gx(mu1, sig1, point):
            fp += 1 #false positive
        else:
            tn += 1 #true negative

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    print('precision: ' + str(precision))
    print('recall: ' + str(recall))
    print('f1: ' + str(f1))


def musigma(data):
    sum = [0.0] * data.shape[1]
    for point in data:
        for i in range(len(point)):
            sum[i] += float(point[i])
    mu = np.array([[x / data.shape[0] for x in sum]])
    data = np.array(data)
    sigma = np.cov(data.transpose())
    return mu, sigma


with open('../data/O_Stars_magnitudes.txt') as f:
    data = np.array([line.split() for line in f], dtype=float)
    o_mu, o_sig = musigma(data)
    o_data = data


with open('../data/M_Stars_magnitudes.txt') as f:
    data = np.array([line.split() for line in f], dtype=float)
    m_mu, m_sig = musigma(data)
    m_data = data

print('Class 1: M stars ------------------')
classify(m_mu, m_sig, o_mu, o_sig, m_data, o_data)
print('Class 2: O stars ------------------')
classify(o_mu, o_sig, m_mu, m_sig, o_data, m_data)