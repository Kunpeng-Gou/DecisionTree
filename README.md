## 数据分析

1. 可以发现这是一个关于一个人是否患有糖尿病的数据集。一共有768人，每个人共有九个特征。

    ```python
    Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    　　　　'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
          dtype='object')
    
    #特征（怀孕次数，血糖，血压，皮脂厚度，胰岛素，BMI身体质量指数，糖尿病遗传函数，年龄，结果）
    ```

2. 这些特征中还包含有缺失值。（下面的计数都是计算的0的个数）

    ```
    Glucose：5    血糖：缺失值较少，可以考虑用平均值填充
    BloodPressure：35    血压：缺失值较少，可以考虑用平均值填充
    SkinThickness： 227    皮脂厚度：缺失值较多，不作填充，在决策树中进一步处理
    Insulin： 374    胰岛素：缺失值较多，不作填充，在决策树中进一步处理
    BMI：11    身体质量指数：缺失值较少，可以考虑用平均值填充
    
    其余数据无缺失值
    ```

3. 可以观察到这些特征都是连续值。这里在决策树中采用二分法进行处理。

4. 初始数据并未划分为训练集和测试集，需要人为进行划分。

    这里采用的是把数据集分割为k个两两不相交的子集。



##决策树构建

1. DecisionTree类中除了构造函数之外，还有其他5个函数。

    子类ID3，C45，CART的choose_best函数均是重新写的，其他4个函数是继承DecionTree的函数

    ```python
    class DecisionTree(object):
        def choose_best(self, node)    # 选取最佳分类特征和最佳切分点
        def build_tree(self, node=None)    # 构建决策树
        def entropy(self, id_list, feature='Outcome', feature_val=0.5)    # 计算信息熵
        def gini(self, id_list, feature='Outcome', feature_val=0.5)    # 计算基尼指数
        def predict(self, test)    # 进行预测
    ```

2. 树的构建方式采用了结点的方法。

    DecisionTree.py中有一个Node类。每个树中有一个root变量储存Node类型的一个变量作为根结点

    ```python
    class Node(object):
        def __init__(self, id_list, feature=None, label=None, feature_val=None):
            self.id_list = id_list
            self.feature = feature
            self.label = label
            self.feature_val = feature_val 
            self.child = {}
        def is_leaf(self)    # 判断结点是否为叶结点
        def feature_ret(self, test)    # 根据返回值判断到达哪一个子结点
        def predict(self, test)    # 进行预测
    ```

    id_list：每个结点中并不储存被划分在该结点的训练数据，而是储存被划分在该结点的训练数据的id

    feature：这是这个结点用来进行判别的特征

    feature_val：这是该特征对应的划分值

    label：如果该结点是叶结点，则储存分类标签

    child：这是一个字典，将  feature_test  的返回值映射到子结点

    ​           

    具体到本例中，feature_test  方法根据  test  的  feature  的值分为    大于 feature_val    和    小于 feature_val

    然后根据返回值（True or False）调用子结点的  predict  方法

    最后到达叶结点，返回该叶结点储存的  label

3. choose_best  方法

    1. 创建三个变量储存全局最佳

        ```python
        best_feature = None  # 全局最佳特征
        global_best_slip = 0  # 全局最佳分割点
        global_best_ret = np.inf  # 全局最佳（的值）
        ```

    2. 遍历特征进行计算（在本例中忽视下面三个特征  （其中'Outcome'是标签）  ['Outcome', 'SkinThickness', 'Insulin']）

    3. 按照西瓜书上所说的二分法计算该特征的分割点

    4. 创建变量储存局部最佳

        ```python
        local_best_slip = 0  # 局部最佳分割点
        local_best_ret = np.inf  # 局部最佳信息增益
        ```

    5. 对每个分割点进行计算，并更新局部最佳

    6. 某个特征的分割点全部计算完成后，更新全局最佳

    7. 重复2 - 6步直到遍历完所有特征

    8. 返回最佳分类特征，最佳分割点，对应的信息增益或信息增益比

