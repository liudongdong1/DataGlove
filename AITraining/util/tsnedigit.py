# """t-SNE对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.transformHandle import fivePointArray
from util.FlexDataRead import DataReadTrain, DataReadTest

data_set = {}
data_set['train']=DataReadTrain(r"../../../data/flexData/digit/digit1","d",fivePointArray)
data_set['val']=DataReadTest(r"../../../data/flexData/digit/digitFlex_7days","d",fivePointArray)
# data_set['test']=FlexSensorDataRead(r"../../data/temp/picFlex/digitGen/test","digit",fivePoint)

dataloader={}
dataloader['train'] = DataLoader(dataset=data_set['train'], batch_size=len(data_set['train']), shuffle=True)
dataloader['val'] = DataLoader(dataset=data_set['val'], batch_size=len(data_set['val']), shuffle=True)
#dataloader['test'] = DataLoader(dataset=data_set['test'], batch_size=CharConfig5Point.BATCHSIZE, shuffle=True)



def get_data():
    dataset=dataloader['val']
    for indexc,data_val in enumerate(dataset):
        inputs,labels = data_val
        print("shape:",inputs.shape,labels.shape)
        return inputs,labels

def get_traindata():
    dataset=dataloader['train']
    for indexc,data_val in enumerate(dataset):
        inputs,labels = data_val
        print("shape:",inputs.shape,labels.shape)
        return inputs,labels
# def plot_embedding(data, label, title):
#     x_min, x_max = np.min(data, 0), np.max(data, 0)
#     data = (data - x_min) / (x_max - x_min)

#     fig = plt.figure()
#     ax = plt.subplot(111)
#     mid = int(data.shape[0] / 2)
#     for i in range(mid):
#         plt.text(data[i, 0], data[i, 1], str("o"),  # 此处的含义是用什么形状绘制当前的数据点
#                  color=plt.cm.Set1(1 / 10),  # 表示颜色
#                  fontdict={'weight': 'bold', 'size': 9})  # 字体和大小
#     for i in range(mid):
#         plt.text(data[mid + i, 0], data[mid + i, 1], str("o"),
#                  color=plt.cm.Set1(2 / 10),
#                  fontdict={'weight': 'bold', 'size': 9})

#     plt.xticks([-0.1, 1.1])  # 坐标轴设置
#     # xticks(locs, [labels], **kwargs)locs，是一个数组或者列表，表示范围和刻度，label表示每个刻度的标签
#     plt.yticks([0, 1.1])
#     plt.title(title)
#     #plt.show()
#     plt.savefig('./tsne.png', dpi=300)


# def main():
#     data, label = get_data()
#     print('Computing t-SNE embedding')
#     tsne = TSNE(n_components=2, init='pca', random_state=0)
#     t0 = time()
#     result = tsne.fit_transform(data)
#     plot_embedding(result, label,
#                    't-SNE embedding of the digits (time %.2fs)'
#                    % (time() - t0))


# if __name__ == '__main__':
#     main()


#--------version2 -------
import numpy as np
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Random state.
RS = 20150101

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
# X,Y=get_data()
# digits_proj = TSNE(random_state=RS).fit_transform(X)

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 26))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.numpy().astype(np.int)])
    #plt.xlim(-25, 25)
    #plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(26):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

# scatter(digits_proj, Y)
# plt.savefig('digits_tsne-generated.png', dpi=120)
# plt.show()


# ------------version 3 -------------
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


X,Y=get_traindata()
X_tsne = TSNE(n_components=2, random_state=33).fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X)

font = {"color": "darkred",
        "size": 13, 
        "family" : "serif"}
font2 = {"color": "darkred",
        "size": 8, 
        "family" : "serif"}
categories=9
plt.style.use("dark_background")
plt.figure(figsize=(8.5, 4))

ax1=plt.subplot(1, 2, 1) 
palette = np.array(sns.color_palette("hls", categories))
#plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=palette[Y.numpy().astype(np.int)], alpha=0.6)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y.numpy(), alpha=0.6, 
             cmap=plt.cm.get_cmap('rainbow', categories))
# txts = []
# for i in range(categories):
#     # Position of each label.
#     xtext, ytext = np.median(X_tsne[Y.numpy() == i, :], axis=0)
#     txt = ax1.text(xtext, ytext, str(i), fontsize=10)
#     txt.set_path_effects([
#         PathEffects.Stroke(linewidth=2, foreground="w"),
#         PathEffects.Normal()])
#     txts.append(txt)

plt.title("t-SNE", fontdict=font)
cbar = plt.colorbar(ticks=range(categories)) 
cbar.set_label(label='digit value', fontdict=font)


#plt.clim(-0.5, 9.5)
ax2=plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y.numpy(), alpha=0.6, cmap=plt.cm.get_cmap('rainbow', categories))
#plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, alpha=0.6)
# txts = []
# for i in range(categories):
#     # Position of each label.
#     xtext, ytext = np.median(X_tsne[Y.numpy() == i, :], axis=0)
#     txt = ax2.text(xtext, ytext, str(i), fontsize=24)
#     txt.set_path_effects([
#         PathEffects.Stroke(linewidth=5, foreground="w"),
#         PathEffects.Normal()])
#     txts.append(txt)

plt.title("PCA", fontdict=font)
cbar = plt.colorbar(ticks=range(categories)) 
cbar.set_label(label='digit value', fontdict=font)
#plt.clim(-0.5, 9.5)
plt.tight_layout()
plt.savefig('digit_train.png')
