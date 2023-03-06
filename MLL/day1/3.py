import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def minmax_demo():
    data = pd.read_csv("datingTestSet2.csv")  # 这个别用txt读进来
    data = data.iloc[:, :3]  # 只取数据前三列
    print("data=\n", data)
    print(type(data))
    transfer = MinMaxScaler()  # 默认0-1
    data_new = transfer.fit_transform(data)
    print("data after MinMaxStandard=\n", data_new)

    transfer2 = StandardScaler()
    data_new2 = transfer2.fit_transform(data)
    print("data after standardScaler\n", data_new2)


if __name__ == '__main__':
    minmax_demo()
