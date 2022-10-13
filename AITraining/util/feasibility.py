import matplotlib.pyplot as plt
import os
import numpy as np
def dataDescription():
    basefolder=r"/home/iot/jupyter/root_dir/liudongdong/data/flexData/chars/picGen/train/"
    fig,axes=plt.subplots(nrows=7,ncols=4,figsize=(96,96))
    #plt.rcParams['font.sans-serif']=['SimHei']
    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus']=False
    # 刻度大小
    plt.rcParams['axes.labelsize']=16
    # 线的粗细
    plt.rcParams['lines.linewidth']=17.5
    # x轴标签大小
    plt.rcParams['xtick.labelsize']=14
    # y轴标签大小
    plt.rcParams['ytick.labelsize']=14
    #图例大小
    plt.rcParams['legend.fontsize']=14
    print(type(axes))
    i=0
    blotplist=[]
    for tempfile in os.listdir(basefolder):
        filename=os.path.join(basefolder,tempfile)
        data=np.loadtxt(filename,delimiter=',').T
        print("label_{}.shape{}. {}".format(tempfile,data.shape[0],data.shape[1]))
        print(i, int(i/4),i%4)
        bplot1=axes[int(i/4),i%4].boxplot(data.tolist(),
                       vert=True,
                       patch_artist=True)
        axes[int(i/4),i%4].yaxis.grid(True) #在y轴上添加网格线
        axes[int(i/4),i%4].set_title(tempfile[0]) #在y轴上添加网格线
        ##axes[int(i/6),(i-int(i/6))%4].set_xticks(["A","B","C","D","E"] ) #指定x轴的轴刻度个数
        axes[int(i/4),i%4].set_xlabel('Finger') #设置x轴名称
        axes[int(i/4),i%4].set_ylabel('Bend Angle') #设置y轴名称
        blotplist.append(bplot1)
        i=i+1
    colors = ['pink', 'lightblue', 'lightgreen','red','orange']
    print("blot length:",len(blotplist))
    for bplot in blotplist:
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    plt.savefig('CharTrain1_fesibility.png', format='png', dpi=100)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FixedLocator

zhfont = matplotlib.font_manager.FontProperties(fname='/home/iot/jupyter/root_dir/liudongdong/src/ai/util/SIMSUN.TTC')

def digitDescription():
    basefolder=r"/home/iot/jupyter/root_dir/liudongdong/data/flexData/digit/digitGen/train"
    fig,axes=plt.subplots(nrows=2,ncols=5,figsize=(60,20))
    #plt.rcParams['font.sans-serif']=['SimHei']
    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus']=False
    # 刻度大小
    plt.rcParams['axes.labelsize']=30
    # 线的粗细
    plt.rcParams['lines.linewidth']=17.5
    # x轴标签大小
    plt.rcParams['xtick.labelsize']=30
    # y轴标签大小
    plt.rcParams['ytick.labelsize']=30
    #图例大小
    plt.rcParams['legend.fontsize']=30
    print(type(axes))
    i=0
    blotplist=[]
    for tempfile in os.listdir(basefolder):
        filename=os.path.join(basefolder,tempfile)
        data=np.loadtxt(filename,delimiter=',').T
        print("label_{}.shape{}. {}".format(tempfile,data.shape[0],data.shape[1]))
        print(i, int(i/5),i%5)
        bplot1=axes[int(i/5),i%5].boxplot(data.tolist(),
                       vert=True,
                       patch_artist=True)
        axes[int(i/5),i%5].yaxis.grid(True) #在y轴上添加网格线
        axes[int(i/5),i%5].set_title(tempfile[0]) #在y轴上添加网格线
        ##axes[int(i/6),(i-int(i/6))%4].set_xticks(["A","B","C","D","E"] ) #指定x轴的轴刻度个数
        axes[int(i/5),i%5].set_xlabel('手指序号',fontproperties=zhfont, fontsize=30, fontweight='bold', labelpad=8.5) #设置x轴名称
        axes[int(i/5),i%5].set_ylabel('弯曲度',fontproperties=zhfont, fontsize=30, fontweight='bold', labelpad=8.5) #设置y轴名称


        #axes[int(i/5),i%5].set_yticklabels(fontproperties='Times New Roman', fontsize=30, fontweight='bold')
        #axes[int(i/5),i%5].set_xlabel(fontproperties=zhfont, fontsize=30, fontweight='bold', labelpad=8.5)
        #axes[int(i/5),i%5].set_ylabel(fontproperties=zhfont, fontsize=30, fontweight='bold', labelpad=8.5)
        plt.grid(axis="y", linestyle='--', linewidth=1.5, color='#e1e2e3', zorder=0)

        axes[int(i/5),i%5].spines['bottom'].set_linewidth(1.5)
        axes[int(i/5),i%5].spines['left'].set_linewidth(1.5)
        axes[int(i/5),i%5].spines['right'].set_linewidth(1.5)
        axes[int(i/5),i%5].spines['top'].set_linewidth(1.5)

        axes[int(i/5),i%5].tick_params(axis='y', length=5, width=0.8)
        axes[int(i/5),i%5].tick_params(axis='x', which='minor', length=0, width=1.5)
        axes[int(i/5),i%5].tick_params(axis='x', which='major', length=0, width=1.5)

        blotplist.append(bplot1)
        i=i+1
    colors = ['pink', 'lightblue', 'lightgreen','red','orange']
    print("blot length:",len(blotplist))
    for bplot in blotplist:
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    plt.savefig('DigitTrain_fesibility.png', format='png', dpi=400,bbox_inches='tight')
#dataDescription()
digitDescription()