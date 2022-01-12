import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# %%
# 设置文件路径
rootdir = 'Pawpularity_Contest/petfinder-pawpularity-score'
filename = 'train.csv'
filedir = rootdir + '/' + filename

# %%
# 读取train.csv
df = pd.read_csv(filedir)

paw = df.Pawpularity.values
print(paw[0:10])
# 将paw中心化
paw = (paw - paw.min())/(paw.max() - paw.min())
# %%
plt.figure(figsize=(20, 20))
plt.hist(paw, bins=10)
plt.show()


# %%
# 训练聚类模型并获得10个分类的label和聚类中心
class Clustering:
    def __init__(self, num_cluster):
        self.num_cluster = num_cluster
        self.labels = None
        self.centers = None

    def fit(self, n=None):
        if n is None:
            n = self.num_cluster
        m_cluster = KMeans(n, random_state=104)
        m_cluster.fit(paw.reshape(-1, 1))
        self.labels = m_cluster.labels_
        self.centers = m_cluster.cluster_centers_


# %%
# 计算不同分类数目下的最小SSE
SSE = []
for i in range(5, 30):
    print('clustering in {} classes'.format(i))
    model = Clustering(i)
    print('Cluster fitting...')
    model.fit()
    labels = model.labels
    centers = model.centers
    paw_cluster = pd.concat([pd.DataFrame(paw), pd.Series(labels)], axis=1)
    paw_cluster.columns = ['Pawpularity', 'labels']
    paw_cluster['after_center'] = centers[paw_cluster['labels']]

    # 计算SSE
    sse = sum((paw_cluster.Pawpularity - paw_cluster.after_center)**2)
    SSE.append(sse)

#%%
# 画图
plt.plot(range(5, 30), SSE)
plt.show()

#%%
# 分成16个类别
model = Clustering(16)
model.fit()
labels = model.labels
centers = model.centers
paw_cluster = pd.concat([pd.DataFrame(paw), pd.Series(labels)], axis=1)
paw_cluster.columns = ['Pawpularity', 'labels']
paw_cluster['after_center'] = centers[paw_cluster['labels']]

# %%
plt.figure(figsize=(20, 20))
plt.hist(paw_cluster.after_center)
plt.show()

#%%
df = pd.concat([df['Id'], df['Pawpularity'], paw_cluster['after_center'], paw_cluster['labels']], axis=1)
df.to_csv('Pawpularity_Contest/petfinder-pawpularity-score/train.csv', index=False)
