import pandas as pd
import random
# rl.to_csv("G:\PythonCode\MLself\Data\data.csv", index=False, sep=',')


def gen_data(row, col, filepath):
    randomlist = []
    for i in range(row*col):
        randomlist.append(random.randint(0,5))
    temp = 1
    mx = []
    my = []
    for i in randomlist:
        if temp <= col:
            mx.append(i)
            temp += 1
        if temp == col+1:
            my.append(mx)
            mx = []
            temp = 1
    # print(my)

    r1 = pd.DataFrame(my)
    # print(r1.shape)
    # print(r1.head(5))

    a = [i for i in range(col)]
    b = ["a"+str(i) for i in range(col)]
    r1.rename(columns=dict(zip(a, b)), inplace=True)
    print(r1.head(5))
    r1.to_csv(filepath, index=False, sep=',')


gen_data(10000, 20, "./data_train.csv")
gen_data(100, 20, "./data_test.csv")
