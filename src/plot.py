import matplotlib.pyplot as plt
import numpy as np
import glob

#pruning features
#same zoom link 11am, 2-5

colors = ['red', 'green', 'blue', 'cyan', 'purple', 'orange', 'magenta']
data_path = "data/*_Stars_magnitudes.txt"
txt_files = glob.glob(data_path)
spTypes = []

for file in txt_files:
    with open(file) as f:
        data = np.array([line.split() for line in f])
    print(data)
    spTypes.append([data.astype(float)])

i = 0
for data in spTypes:
    data = np.array(data)[0]
    print(data.shape)
    print(data)
    x,y,z = data.T
    x = x.squeeze()
    y = y.squeeze()
    z = z.squeeze()
    diff = x.copy()
    for j in range(len(x)):
        diff[j] = float(x[j]-y[j])
    #print(x)
    plt.scatter(diff,z, c=colors[i])
    i += 1
plt.xticks(np.arange(0, 30, step=3))
plt.yticks(np.arange(0, 30, step=3))
plt.show()


'''
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Magnitude')
ax1.set_xlabel('Spectral Type')
x = np.array([1,2,3,4,5,6])
y = np.array([-10,-5,0,5,10,15])
my_xticks = ['B', 'A', 'F', 'G', 'K', 'M']
my_yticks = [15, 10, 5, 0, -5, -10]
plt.plot(x,y)
plt.xticks(x, my_xticks)
plt.yticks(y, my_yticks)

plt.show()
'''