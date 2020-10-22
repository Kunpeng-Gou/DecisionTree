import data_processing as dp


class Node(object):
    def __init__(self):
        pass

class DecisionTree(object):
    def __init__(self, algor=None, thresh=(0.01, 10), thresh_val=0.5, train_data=None, root=None):
        self.algor = algor    #算法名称
        self.thresh = thresh    #前剪枝的一些阈值
        self.thresh_val = thresh_val    #决定分类的阈值
        self.train_data = train_data    #训练集
        self.root = root    #根节点

    def predict(self):
        print("DT predict")
        return 0

    def build_tree(self):
        pass


class ID3(DecisionTree):
    def __init__(self, algor='ID3', thresh=(0.0001, 8), thresh_val=0.5,):
        #super(ID3, self).__init__()
        self.algor = algor
        self.thresh = thresh  # 前剪枝的一些阈值
        self.thresh_val = thresh_val  # 决定分类的阈值


if __name__ == '__main__':
    tree = ID3()
    tree.predict()
    print(tree.algor)


    print('finished')
