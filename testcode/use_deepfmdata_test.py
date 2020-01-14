# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 15:45:51 2019

@author: Administrator
"""
from testcode.XdeepFM_batch_next_test import BacthDataset


label = [1, 2, 3, 4, 5, 6, 7, 8, 9]
xi = [[1, 1, 1], [2, 2, 2], [3, 3, 3],
      [4, 4, 4], [5, 5, 5], [6, 6, 6],
      [7, 7, 7], [8, 8, 8], [9, 9, 9]]
xv = [[1, 0, 1], [2, 0, 2], [3, 0, 3],
      [4, 0, 4], [5, 0, 5], [6, 6, 6],
      [7, 0, 7], [8, 0, 8], [9, 0, 9]]
dataset = BacthDataset(xi, xv, label)

i = 20
cnt = 0
while i > 0:
    cnt += 1
    if cnt % 3 == 0:
        print("当前轮数: %d" % (dataset._epochs_completed))
    print(dataset.next_batch(4))
    i -= 1
# 样本数量
print(dataset.num_examples)


# xi_train, xi_valid, xv_train, xv_valid, y_train, y_valid = dataset.train_test_split(test_size=0.3)
# print(xi_train)
# print(xi_valid)
# print(xv_train)
# print(xv_valid)
# print(y_train)
# print(y_valid)
# print("测试集长:%d" % (dataset.test_lens))
# print("训练集长:%d" % (dataset.train_lens))






# 测试数据完毕  测试数据正常
