import matplotlib.pyplot as plt
import numpy as np

def remove_nan(lst, i):
    return lst[[not b for b in np.isnan(lst[:, i])], :]

columns = ["Life Ladder", "Log GDP per capita", "Social support", "Healthy life expectancy at birth",
          "Freedom to make life choices", "Generosity", "Perceptions of corruption", "Confidence in national government",
          "Democratic Quality", "Delivery Quality", "Class"]

# read csv file
xy = np.genfromtxt('dataset/all_data.csv', delimiter=',', skip_header=1)

group1 = xy[xy[:,-1] == 0, :]
group2 = xy[xy[:,-1] == 1, :]
group3 = xy[xy[:,-1] == 2, :]

p1 = []
p2 = []
for i in range(1, xy.shape[1]-1):
    set1 = remove_nan(group1, i)
    set2 = remove_nan(group2, i)
    set3 = remove_nan(group3, i)
    n = 0
    m = 0
    a = set1.shape[0]
    b = set2.shape[0]
    c = set3.shape[0]
    for x in range(a):
        for y in range(b):
            for z in range(c):
                if set1[x][i] >= set2[y][i] >= set3[z][i]:
                    n += 1
                elif set1[x][i] <= set2[y][i] <= set3[z][i]:
                    m += 1
    p1 += [n / (a * b * c)]
    p2 += [m / (a * b * c)]
    plt.hist([set1[:, i], set2[:, i], set3[:, i]])
    plt.xlabel(columns[i])
    # plt.ylabel(columns[0])
    plt.savefig('img/histogram' + str(i) + '.png')
    plt.show()
print(p1)
print(p2)
# Result
# [0.6867686319548126, 0.6224133975088485, 0.6533456060305495, 0.4079433529270508, 0.20141978513035216, 0.07199396945050585, 0.10523011516094342, 0.5064657764228067, 0.5528475010179684, 0.0]
# [0.0031926497457689055, 0.005749201808329592, 0.004132296223598075, 0.03257196671504192, 0.10498094154250932, 0.2706418567744495, 0.17524345837814134, 0.020179638542895624, 0.014979450610259034, 1.0]
# [0.687, 0.622, 0.653, 0.408, 0.201, 0.072, 0.105, 0.506, 0.553]
# [0.003, 0.006, 0.004, 0.033, 0.105, 0.271, 0.175, 0.020, 0.015]



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