# k-means 聚类
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from matplotlib import pyplot
# 定义数据集
X, _ = make_classification(n_samples=100,               #数据的数量
                           n_features=2, 
                           n_informative=2,
                           n_redundant=0, 
                           n_clusters_per_class=1, 
                           random_state=3)              #随机种子，不同的种子生成不同的数据集

# 定义模型
model = KMeans(n_clusters=3)                            #聚类所划分的群体个数

# 模型拟合
model.fit(X)

# 为每个示例分配一个集群
yhat = model.predict(X)

# 检索唯一群集
clusters = unique(yhat)
250
# 为每个群集的样本创建散点图
for cluster in clusters:
    # 获取此群集的示例的行索引
    row_ix = where(yhat == cluster)
    # 创建这些样本的散布
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# 绘制散点图
pyplot.show()
