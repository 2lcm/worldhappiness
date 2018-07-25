import matplotlib.pyplot as plt
import numpy as np

def remove_nan(lst, i):
    return lst[[not b for b in np.isnan(lst[:, i])], :]

# read csv file
xy = np.genfromtxt('dataset/contribution_data.csv', delimiter=',', skip_header=1)

group1 = xy[xy[:,-2] == 0, :]
group2 = xy[xy[:,-2] == 1, :]
t1 = int(group1.shape[0] * 0.9)
t2 = int(group2.shape[0] * 0.9)
train_set1 = group1[:t1, :]
train_set2 = group2[:t2, :]
test_set1 = group1[t1:, :]
test_set2 = group2[t2:, :]

p = []
for i in range(1, xy.shape[1]):
    train_set1_tmp = remove_nan(train_set1, i)
    train_set2_tmp = remove_nan(train_set2, i)
    n = 0
    a = train_set1_tmp.shape[0]
    b = train_set2_tmp.shape[0]
    for x in range(train_set1_tmp.shape[0]):
        for y in range(train_set1_tmp.shape[0]):
            if train_set1_tmp[x][i] > train_set2_tmp[y][i]:
                n += 1
    p += [n / (a * b)]
    plt.hist([train_set1_tmp[:, i], train_set2_tmp[:, i]], label=['group1', 'group2'])
    plt.legend(loc='upper right')
    plt.show()
print(p)

########################################################################################################

# p = []
# for i in range(1, xy.shape[1]-2):
#     n = 0
#     for a in range(group1.shape[0]):
#         for b in range(group2.shape[0]):
#             if group1[a][i] >= group2[b][i]:
#                 n += 1
#     p += [n / (a * b)]
#
# print("Class:", p)
#
# p = []
# for i in range(1, xy.shape[1]-2):
#     n = 0
#     print('i=', i)
#     for a in range(group21.shape[0]):
#         for b in range(group22.shape[0]):
#             for c in range(group23.shape[0]):
#                 if group21[a][i] >= group22[b][i] >= group23[c][i]:
#                     n += 1
#     p += [n / (a * b * c)]
# print("Class2:", p)
# [0.708222695286206, 0.4893587666964747, 0.6542644568423804, 0.4318145829746486, 0.24192071143090596, 0.21058716427828217, 0.3987900975948785]