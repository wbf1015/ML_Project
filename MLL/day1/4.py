import pandas as pd
from sklearn.feature_selection import VarianceThreshold

'''
一种给特征降维的方式，这是最简单的，直接删掉方差最小的
可以参考的博客
https://blog.csdn.net/weixin_45901519/article/details/114685227
'''


def variance_demo():
    """
    删除低方差特征——特征选择
    :return: None
    """
    data = pd.read_csv("factor_returns.csv")
    print(data)
    print(type(data))
    # 1、实例化一个转换器类
    transfer = VarianceThreshold(threshold=50)  # 指定阀值方差 低于该方差的特征会被删除
    # 2、调用fit_transform
    data = transfer.fit_transform(data.iloc[:, 1:10])
    print("删除低方差特征的结果：\n", data)
    print("形状：\n", data.shape)

    return None


if __name__ == '__main__':
    variance_demo()
