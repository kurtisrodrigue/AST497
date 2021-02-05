import numpy as np

def gx(mu, sig, point):
    point = np.array([point])
    W = np.matmul((mu-point), sig.transpose())
    value = - 0.5 * np.matmul(W, (mu-point).transpose())
    value -= -0.5 * np.log(np.linalg.det(sig))
    return value


def classify(mu1, sig1, mu2, sig2, datac1, datac2):
    c1_miss = c1_correct = 0
    c2_miss = c2_correct = 0
    for point in datac1:
        if gx(mu1, sig1, point) < gx(mu2, sig2, point):
            c1_miss += 1
        else:
            c1_correct += 1
    for point in datac2:
        if gx(mu2, sig2, point) < gx(mu1, sig1, point):
            c2_miss += 1
        else:
            c2_correct += 1

    print('class 1: ' + str(c1_miss) + ' misses, ' + str(c1_correct) + ' correct')
    print(c1_correct / (c1_miss +c1_correct))
    print('class 2: ' + str(c2_miss) + ' misses, ' + str(c2_correct) + ' correct')
    print(c2_correct / (c2_miss + c2_correct))


def musigma(data):
    sum = [0.0] * data.shape[1]
    print(str(data.shape) + '\n')
    for point in data:
        for i in range(len(point)):
            sum[i] += float(point[i])
    mu = np.array([[x / data.shape[0] for x in sum]])
    data = np.array(data)
    print(data.dtype)
    sigma = np.cov(data.transpose())
    print(mu, sigma)
    return mu, sigma


with open('../data/O_Stars_magnitudes.txt') as f:
    data = np.array([line.split() for line in f], dtype=float)
    print(data)
    o_mu, o_sig = musigma(data)
    o_data = data


with open('../data/M_Stars_magnitudes.txt') as f:
    data = np.array([line.split() for line in f], dtype=float)
    m_mu, m_sig = musigma(data)
    m_data = data

classify(m_mu, m_sig, o_mu, o_sig, m_data, o_data)