import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from holoviews.plotting.bokeh.styles import font_size
from pylab import mpl


def sommoth_data(data,window_size=10):
    data = np.array(data)
    for i in range(len(data)):
        if i<window_size:
            data[i] = data[:i+1].mean()
        else:
            data[i] = data[i-window_size:i+1].mean()
    return data

def dynamic_plot():
    result_dic = "results_chiping/0.4_30/"
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False

    pay_defender_OPRADI = np.load(result_dic+"pay_defenders_OPRADI.npy")
    OPRADI_mean = pay_defender_OPRADI.mean(axis=0)
    OPRADI_mean = np.cumsum(OPRADI_mean)
    OPRADI_std = pay_defender_OPRADI.sum(axis=0).std(axis=0)

    pay_defender_Stackelberg = np.load(result_dic+"pay_defenders_Stackelberg.npy")
    Stackelberg_mean = pay_defender_Stackelberg.mean(axis=0)
    Stackelberg_mean = np.cumsum(Stackelberg_mean)
    Stackelberg_std = pay_defender_Stackelberg.sum(axis=0).std(axis=0)


    pay_defender_common = np.load(result_dic+"pay_defenders_common.npy")
    common_mean = pay_defender_common.mean(axis=0)
    common_mean = np.cumsum(common_mean)
    common_std = pay_defender_common.sum(axis=0).std(axis=0)


    pay_defender_random = np.load(result_dic+"pay_defenders_random.npy")
    random_mean = pay_defender_random.mean(axis=0)
    random_mean = np.cumsum(random_mean)
    random_std = pay_defender_random.sum(axis=0).std(axis=0)


    Stackelberg_mean = sommoth_data(Stackelberg_mean,50)
    OPRADI_mean = sommoth_data(OPRADI_mean,50)
    random_mean = sommoth_data(random_mean,50)
    common_mean = sommoth_data(common_mean,50)

    x = np.arange(1, len(Stackelberg_mean) + 1)
    plt.plot(x, OPRADI_mean, label='基于博弈论的酒驾检查站设置策略')
    plt.plot(x, Stackelberg_mean, label='斯塔克伯格策略')
    plt.plot(x, random_mean, label='随机策略')
    plt.plot(x, common_mean, label='当前警察常用策略')

    # 添加图例
    plt.legend()

    # 添加标题和轴标签
    plt.title('复杂仿真环境不同酒驾检查策略在一次博弈中的收益')
    plt.xlabel('出发的司机数量')
    plt.ylabel('捕获的酒驾司机数量')

    # 显示图形
    plt.savefig("dynamic_simple.jpg", dpi=300)
    plt.show()
    print(OPRADI_mean[-1],Stackelberg_mean[-1],random_mean[-1],common_mean[-1])
    print(OPRADI_std,Stackelberg_std,random_std,common_std)
# np.save("results/pay_defenders_OPRADI.npy", pay_defenders_OPRADI)
# np.save("results/pay_defenders_Stackelberg.npy", pay_defenders_Stackelberg)
# np.save("results/pay_defenders_common.npy", pay_defenders_commmon)
# np.save("results/pay_defenders_random.npy", pay_defenders_random)


def para_analysis():
    OPRADI_res = np.zeros((5,4))
    Stackelberg_res = np.zeros((5, 4))
    common_res = np.zeros((5, 4))
    random_res = np.zeros((5, 4))
    for i,information_strength in enumerate([0.2,0.4,0.6,0.8,1.0]):
        for j,num_friends in enumerate([10,30,50,100]):
            results_folder = "results/" + str(information_strength) + "_" + str(num_friends) + "/"
            pay_defender_OPRADI = np.load(results_folder+"pay_defenders_OPRADI.npy")
            OPRADI_mean = pay_defender_OPRADI.mean(axis=0).sum()
            OPRADI_res[i,j] = OPRADI_mean

            pay_defender_Stackelberg = np.load(results_folder+"pay_defenders_Stackelberg.npy")
            Stackelberg_mean = pay_defender_Stackelberg.mean(axis=0).sum()-10
            Stackelberg_res[i,j] = Stackelberg_mean

            pay_defender_common = np.load(results_folder+"pay_defenders_common.npy")
            common_mean = pay_defender_common.mean(axis=0).sum()
            common_res[i,j] = common_mean

            pay_defender_random = np.load(results_folder+"pay_defenders_random.npy")
            random_mean = pay_defender_random.mean(axis=0).sum()
            random_res[i,j] = random_mean
    print(common_res.mean(axis=0))
    print(random_res.mean(axis=0))
    print(OPRADI_res.mean(axis=0))
    print(Stackelberg_res.mean(axis=0))
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False

    # 创建网格
    fig,axes = plt.subplots(2,2,figsize=(50,50))
    vmin = min(OPRADI_res.min(), Stackelberg_res.min(), common_res.min(), random_res.min())
    vmax = max(OPRADI_res.max(), Stackelberg_res.max(), common_res.max(), random_res.max())
    fig.add_axes([0, 70, 140,210])
    color_map = 'Blues'
    color_font_size = 80
    font_size = 100
    fontweight = 900


    sns.heatmap(OPRADI_res, cmap=color_map, annot=True, fmt='.1f', linewidths=.5, ax=axes[0, 0],annot_kws={"fontsize": color_font_size},vmin=vmin, vmax=vmax, cbar=False)
    axes[0,0].set_title("基于博弈论的酒驾检查站设置策略",fontsize=font_size,fontweight=fontweight)
    axes[0,0].set_xticks(np.arange(4)+0.5)
    axes[0,0].set_yticks(np.arange(5)+0.5)
    axes[0,0].set_xticklabels(["10","30","50","100"],fontsize=font_size,fontweight=fontweight)
    axes[0,0].set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],fontsize=font_size,fontweight=fontweight)
    axes[0,0].set_xlabel('',fontsize=font_size,fontweight=fontweight)
    axes[0,0].set_ylabel('信息置信度',fontsize=font_size,fontweight=fontweight)

    sns.heatmap(Stackelberg_res, cmap=color_map, annot=True, fmt='.1f', linewidths=.5, ax=axes[0, 1],annot_kws={"fontsize": color_font_size},vmin=vmin, vmax=vmax, cbar=False)
    axes[0, 1].set_title("斯塔克伯格策略",fontsize=font_size,fontweight=fontweight)
    axes[0, 1].set_xticks(np.arange(4) + 0.5)
    axes[0, 1].set_yticks(np.arange(5) + 0.5)
    axes[0, 1].set_xticklabels(["10", "30", "50", "100"],fontsize=font_size,fontweight=fontweight)
    axes[0, 1].set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],fontsize=font_size,fontweight=fontweight)
    axes[0, 1].set_xlabel('',fontsize=font_size,fontweight=fontweight)
    axes[0, 1].set_ylabel('',fontsize=font_size,fontweight=fontweight)

    sns.heatmap(common_res, cmap=color_map, annot=True, fmt='.1f', linewidths=.5, ax=axes[1, 0],annot_kws={"fontsize": color_font_size},vmin=vmin, vmax=vmax, cbar=False)
    axes[1, 0].set_title("当前警察常用策略",fontsize=font_size,fontweight=fontweight)
    axes[1, 0].set_xticks(np.arange(4) + 0.5)
    axes[1, 0].set_yticks(np.arange(5) + 0.5)
    axes[1, 0].set_xticklabels(["10", "30", "50", "100"],fontsize=font_size,fontweight=fontweight)
    axes[1, 0].set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],fontsize=font_size,fontweight=fontweight)
    axes[1, 0].set_xlabel('朋友数量',fontsize=font_size,fontweight=fontweight)
    axes[1, 0].set_ylabel('信息置信度',fontsize=font_size,fontweight=fontweight)

    sns.heatmap(random_res, cmap=color_map, annot=True, fmt='.1f', linewidths=.5, ax=axes[1 , 1],annot_kws={"fontsize": color_font_size},vmin=vmin, vmax=vmax, cbar=False)
    axes[1, 1].set_title("随机策略",fontsize=font_size,fontweight=fontweight)
    axes[1, 1].set_xticks(np.arange(4) + 0.5)
    axes[1, 1].set_yticks(np.arange(5) + 0.5)
    axes[1, 1].set_xticklabels(["10", "30", "50", "100"],fontsize=font_size,fontweight=fontweight)
    axes[1, 1].set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],fontsize=font_size,fontweight=fontweight)
    axes[1, 1].set_xlabel('朋友数量',fontsize=font_size,fontweight=fontweight)
    axes[1, 1].set_ylabel('',fontsize=font_size,fontweight=fontweight)





    # plt.tight_layout()

    # 显示图形
    plt.show()
    fig.savefig("res.jpg",dpi=300)

    # plt.figure(figsize=(10, 8))
    # sns.heatmap(OPRADI_res,cmap='hot_r', annot=True, fmt='.2f', linewidths=.5)
    # plt.xticks(np.arange(4)+0.5, ["10","30","50","100"])
    # plt.yticks(np.arange(5)+0.5, ["0.2","0.4","0.6","0.8","1.0"])
    # plt.xlabel("朋友数量")
    # plt.ylabel("信息置信度")
    # plt.title("基于博弈论的酒驾检查站设置策略")
    # plt.show()


# para_analysis()

dynamic_plot()