import numpy as np
import matplotlib.pyplot as plt

dataset = np.genfromtxt('dataset/data2018.csv', delimiter=',', skip_header=1)
dataset = dataset[:, 0]
y_data = [0, 0, 0, 0, 0, 0, 0]
x_data = [i+2 for i in range(len(y_data))]

for dt in dataset:
    y_data[int(dt) - 2] += 1

print(y_data)
plt.hist(dataset, bins=x_data, rwidth=0.8)
tmp = [b + 0.2 for b in x_data]
p = ["{:.1f}%".format(y / sum(y_data) * 100) for y in y_data]
plt.xlabel('Happiness Score')
plt.savefig('img/score_histogram.png')
# plt.plot(x_data, y_data)
plt.show()
print(p)

