import warnings

warnings.filterwarnings('ignore')
from sklearn import datasets  # 导入库
from sklearn import model_selection
from sklearn import feature_extraction

'''
可能可以参考的：
https://zhuanlan.zhihu.com/p/108393576

'''


def loadBoston():
    boston = datasets.load_boston()  # 导入波士顿房价数据
    print(boston.keys())  # 查看键(属性)     ['data','target','feature_names','DESCR', 'filename']
    print(boston.data.shape, boston.target.shape)  # 查看数据的形状 (506, 13) (506,)
    print(boston.feature_names)  # 查看有哪些特征 这里共13种
    print(boston.DESCR)  # described 描述这个数据集的信息
    print(boston.filename)  # 文件路径


def loadIris():
    iris = datasets.load_iris()  # 导入鸢尾花数据
    print(iris.data.shape, iris.target.shape)  # (150, 4) (150,)
    print(iris.feature_names)  # [花萼长，花萼宽，花瓣长，花瓣宽]


def course_load_data():
    iris = datasets.load_iris()
    print("鸢尾花数据集:\n", iris)
    print("查看数据集描述:\n", iris["DESCR"])
    print("查看特征值的名字:\n", iris.feature_names)
    print("查看特征值:\n", iris.data)
    print("查看特征值形状:\n", iris.data.shape)

    # 数据集的划分
    x_train, x_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.2,
                                                                        random_state=22)
    print("训练集的特征值:\n", x_train, x_train.shape)
    print("测试集的特征值:\n", x_test, x_test.shape)

    return None


'''
字典特征提取
作用：对字典数据进行特征值化
本部分可以参考
https://blog.csdn.net/weixin_44517301/article/details/88405939
'''


def course_process_data():
    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60},
            {'city': '深圳', 'temperature': 30}]
    # 实例一个转换器类
    transfer = feature_extraction.DictVectorizer(sparse=False)
    # 调用fit_transform（）
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    print("特征名称:\n", transfer.get_feature_names())

    # spare这玩意就是一个稀疏矩阵,标志了哪个地方是非零值，非零值是多少
    transfer2 = feature_extraction.DictVectorizer()
    data_new = transfer2.fit_transform(data)
    print("data_new:\n", data_new)
    print("特征名称:\n", transfer2.get_feature_names())
    return None


'''
文本特征提取
作用：对文本数据进行特征值化
如果输出为：(0, 2)	1
可以理解为在第零个字符串中，单词标号为2的单词总共出现了1次
可以参考的：
https://blog.csdn.net/qq_27328197/article/details/113811917
'''


def engchip():
    data = {"Life is short,i like like python", "Life is too long,i dislike python"}
    # 实例化一个转化器
    transfer = feature_extraction.text.CountVectorizer()
    # 调用transform
    data_new = transfer.fit_transform(data)
    print("字符与其参数对应\n", transfer.vocabulary_)
    print("data_new:\n", data_new)
    print("data_new:\n", data_new.toarray())
    # print("特征名字:\n", transfer.get_feature_names())
    return None


'''
对中文进行分词处理
'''


def count_chinese_demo():
    """
    文本特征抽取：CountVecotrizer统计每个样本特征词出现的次数
    不手动空格的话，需要jieba分词先处理才能得到单词
    """
    data = ["五星红旗 我 为你 骄傲", "五星红旗 我 为你 自豪"]
    # 1、实例化一个转换器类
    transfer = feature_extraction.text.CountVectorizer()

    # 2、调用fit_transform,用toarray()来显示数组
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())
    print("feature_name:\n", transfer.get_feature_names())

    return None


'''
可能需要参考的博客：
https://blog.csdn.net/codejas/article/details/80356544
'''

import jieba


def count_chinese_demo2():
    '''
    中文文本特征抽取，自动分词
    :return:
    '''
    data = ["吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮",
            "若要不吃葡萄非吐皮,就得先吃葡萄不吐皮",
            "青葡萄,紫葡萄,青葡萄没紫葡萄紫"]
    data_new = []
    for sent in data:
        # list()强转为列表形式
        # “ ”.join()强转为字符串格式,其中空格是分词符号
        data_new.append(" ".join(jieba.lcut(sent)))

    # 实例化一个转换器类
    transfer = feature_extraction.text.CountVectorizer()

    # 调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print(type(data_final))
    print("data:\n", data_final.toarray())
    print("特征名字：\n", transfer.get_feature_names())

    return None


def cut_word(text):
    return " ".join(list(jieba.cut(text)))


def count_chinese_demo3():
    data = {"在北上广深，软考证书可以混个工作居住证，也是一项大的积分落户筹码。",
            "升职加薪必备，很多企业人力资源会以此作为审核晋升的条件。",
            "简历上浓彩重抹一笔，毕竟是国家人力部、工信部承认的IT高级人才。"}
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # 实例化一个转化器
    transfer = feature_extraction.text.CountVectorizer()
    # 调用transform
    data_final = transfer.fit_transform(data_new)
    print("data_new:\n", data_final.toarray())
    print("特征名字:\n", transfer.get_feature_names())
    return None


if __name__ == '__main__':
    count_chinese_demo3()
