import warnings

warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer

'''
文本向量化的一个例子
https://blog.csdn.net/qq_43391414/article/details/112912107
'''


def example_1():
    # 训练数据
    train = ["Chinese Beijing Chinese", "Chinese Chinese Shanghai", "Chinese Macao", "Tokyo Japan Chinese"]

    '''
    第一步 填充tx
    '''
    # 将训练数据转化为向量。
    tv = TfidfVectorizer()  # 初始化一个空的tv。
    tv_fit = tv.fit_transform(train)  # 用训练数据充实tv,也充实了tv_fit。
    print("fit后，所有的词汇如下：")
    print(tv.get_feature_names())
    print("fit后，训练数据的向量化表示为：")
    print(tv_fit.toarray())

    '''
    第二步，填充tv_test
    '''
    test = ["Chinese Beijing shanghai"]
    tv_test = tv.transform(test)  # 测试数据不会充实或者改变tv,但这步充实了tv_test。
    print("所有的词汇如下：")
    print(tv.get_feature_names())
    print("测试数据的向量化表示为：")
    print(tv_test.toarray())


'''
停用词，停用词就是说一些无关紧要的词，比如中文中{"的“，”地“}等等。你可以提供一个停用词的库给tv，
那么tv将在文档中自动忽略这些停用词，相当于对文档做了一个预处理，删除了这些文档中的所有停用词
'''


def example_2():
    # 训练数据
    train = ["Chinese Beijing Chinese", "Chinese Chinese Shanghai", "Chinese Macao", "Tokyo Japan Chinese"]

    # 将训练数据转化为向量。

    tv = TfidfVectorizer(stop_words=["chinese"])  # 停用词注意要用小写，因为train会被自动转成小写。
    tv_fit = tv.fit_transform(train)  # 用训练数据充实tv,也充实了tv_fit。
    print("fit后，所有的词汇如下：")
    print(tv.get_feature_names())


'''
这表示，sklearn中内部有一个大家普遍都认同的英语停用词库，比如"the"等。注意这是sklearn内置的，中文没有。
'''


def example_3():
    # 训练数据
    train = ["the Chinese Beijing Chinese", "Chinese Chinese Shanghai", "Chinese Macao", "Tokyo Japan Chinese"]

    # 将训练数据转化为向量。

    tv = TfidfVectorizer(stop_words="english")  # 初始化一个空的tv。
    tv_fit = tv.fit_transform(train)  # 用训练数据充实tv,也充实了tv_fit。
    print("fit后，所有的词汇如下：")
    print(tv.get_feature_names())


'''
我们发现，比之前多了一些词汇，现在两个单词组合在一起也被认为是一个词汇了。
这是自然语言处理中的2元短语。在此处，这个参数表示将1元短语（单词），2元短语都看作总词汇表中的1项。
'''


def example_4():
    # 训练数据
    train = ["Chinese Beijing Chinese", "Chinese Chinese Shanghai", "Chinese Macao", "Tokyo Japan Chinese"]

    # 将训练数据转化为向量。
    tv = TfidfVectorizer(ngram_range=(1, 2))  # 初始化一个空的tv。
    tv_fit = tv.fit_transform(train)  # 用训练数据充实tv,也充实了tv_fit。
    print("fit后，所有的词汇如下：")
    print(tv.get_feature_names())
    print("fit后，训练数据的向量化表示为：")
    print(tv_fit.toarray())


'''
这个df应该还记得把？就是文档频率，但是注意不是逆（倒数）的，比如在上面"Shanghai"出现在4篇文档中的1篇，那么其频率就是0.25。
这个参数的意思就是删去那些在90%以上的文档中都会出现的词，同时也删去那些没有出现在至少2篇文档中的词。
'''


def example_5():
    # 训练数据
    train = ["Chinese Beijing Chinese", "Chinese Chinese Shanghai", "Chinese Macao", "Tokyo Japan Chinese"]

    # 将训练数据转化为向量。

    tv = TfidfVectorizer(max_df=0.9, min_df=0.25)  # 初始化一个空的tv。
    tv_fit = tv.fit_transform(train)  # 用训练数据充实tv,也充实了tv_fit。
    print("fit后，所有的词汇如下：")
    print(tv.get_feature_names())


'''
vocabulary=[]，指明你想要捕获的单词，其实相当于指定了一个文档的向量维度。
注意一下一个坑：默认的话所有文档会转成小写，所以你指定的vocabulary得是小写的单词。否则算出来的向量全都是零
'''


def example_6():
    # 训练数据
    train = ["Chinese Beijing Chinese", "Chinese Chinese Shanghai", "Chinese Macao", "Tokyo Japan Chinese"]

    # 将训练数据转化为向量。
    v = ["chinese", "beijing"]  # 必须全都转成小写
    tv = TfidfVectorizer(vocabulary=v)  # 初始化一个空的tv。
    tv_fit = tv.fit_transform(train)  # 用训练数据充实tv,也充实了tv_fit。
    print("fit后，所有的词汇如下：")
    print(tv.get_feature_names())
    print("fit后，训练数据的向量化表示为：")
    print(tv_fit.toarray())


'''
这个可以将文档在字符级别转成向量，平常都是单词级别转成向量。
'''


def example_7():
    # 标签是字符串
    a = ['h', "e", "l", "l", "oe"]
    atv = TfidfVectorizer(analyzer="char")  # 以char为单位进行分词
    atv_fit = atv.fit_transform(a)

    # 下面这行代码是打印标签对应哪一列为1，这个TfidfVectorizer是按字母顺序排序的a-z。
    print(atv.get_feature_names())
    av = atv_fit.toarray()


if __name__ == '__main__':
    example_7()
