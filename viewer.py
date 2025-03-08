import streamlit as st
import pandas as pd
import numpy as np
import os
import argparse
from heapq import heappush, heappop
from dui_scenario import scenario
from itertools import count
import matplotlib.pyplot as plt
import networkx as nx
import ast

# 设置页面标题
st.title('酒驾检查站设置策略展示')
# 创建一个侧边栏
st.sidebar.header('地图信息初始化')

# 在侧边栏中添加滑动条用于选择数据数量
num_polices = st.sidebar.slider('检查点数量', min_value=1, max_value=10, value=3, step=1)
num_drivers = st.sidebar.slider('司机数量', min_value=100, max_value=10000, value=500, step=100)
num_duidrivers = st.sidebar.slider('酒驾司机数', min_value=10, max_value=1000, value=50, step=10)
n_rows = st.sidebar.slider('行数', min_value=3, max_value=20, value=5, step=1)
n_columns = st.sidebar.slider('列数', min_value=3, max_value=20, value=5, step=1)
knowledge_strength = st.sidebar.slider('知识传播强度', min_value=0, max_value=10, value=3, step=1)
iter_times = st.sidebar.slider('迭代次数', min_value=0, max_value=50, value=2, step=1)
# 输入酒驾司机出发的节点位置
source_region = st.sidebar.text_input("输入酒驾司机经常出发的节点位置", value="1")
del_node_list = st.sidebar.text_input("输入应该删除的节点编号", value="")
del_edge_list = st.sidebar.text_input("输入应该删除的边的节点对", value="")

# 创建一个场景对象
dui_object = scenario(knowledge_strength, 30, num_polices, num_drivers, num_duidrivers, 1, n_rows, n_columns)
# 计算并显示抽象地图
if len(del_node_list)>0:
    del_node_list = [int(x.strip()) for x in del_node_list.split(',')]
    
if len(del_edge_list)>0:
    split_res = ast.literal_eval(f"[{del_edge_list}]")
    del_edge_list = split_res
    # print(split_res)
    # del_edge_list = [tuple(item) if isinstance(item, tuple) else item for item in split_res[0]]
    # print(del_edge_list)
abstract_map = dui_object.mapinit(n_rows, n_columns,del_node_list,del_edge_list)


# 画出地图
fig, ax = plt.subplots(figsize=(15, 15))
array = [int(x.strip()) for x in source_region.split(',')]  # 将输入的源区域转换为整数列表

# 绘制地图和源区域
nx.draw(abstract_map, pos=dui_object.pos, ax=ax)  # 绘制整体地图
nx.draw(abstract_map, pos=dui_object.pos, ax=ax, node_size=500, node_color='skyblue', with_labels=True)
nx.draw_networkx_labels(abstract_map, pos=dui_object.pos, ax=ax, font_size=10, font_color='black') # 在图中显示节点编号
nx.draw(abstract_map, pos=dui_object.pos, nodelist=array, node_size=1000,node_color='r', ax=ax)  # 标记源区域（红色节点）

option = st.radio("",("初始化地图抽象展示", "酒驾检查站设置策略展示"))

st.markdown(
    """
    <style>
    div.row-widget.stRadio > div {
        flex-direction: row;
    }
    div.stRadio > div[role="radiogroup"] label {
        font-size: 100px; /* 增大字体大小 */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# 根据选择展示不同的图形
if option == "初始化地图抽象展示":
    # st.write("### 地图抽象")
    st.pyplot(fig)
elif option == "酒驾检查站设置策略展示":
    # st.write("### 推荐检查点设置")
    # 可选：显示预期检查到的酒驾司机数
    y, x = dui_object.run(max_iter=iter_times)

    sorted_index = np.argsort(y)

    reco_edges = []
    target_list = sorted_index[:num_polices + 1]
    for target in target_list:
        found = 0
        for i in range(len(dui_object.map_index)):
            for j in range(len(dui_object.map_index[0])):
                if dui_object.map_index[i][j] == target:
                    reco_edges.append((i, j))
                    found = 1
                if found == 1:
                    break
            if found == 1:
                break
    # 可选：显示推荐检查点设置
    figr, axr = plt.subplots(figsize=(10, 10))

    nx.draw(abstract_map, pos=dui_object.pos, ax=axr)  # 绘制整体地图
    nx.draw(abstract_map, pos=dui_object.pos, ax=axr, node_size=500, node_color='skyblue', with_labels=True)
    nx.draw_networkx_labels(abstract_map, pos=dui_object.pos, ax=axr, font_size=10, font_color='black')  # 在图中显示节点编号
    nx.draw(abstract_map, pos=dui_object.pos, nodelist=array, node_size=1000, node_color='r', ax=axr)
    nx.draw(abstract_map, pos=dui_object.pos, edgelist=reco_edges, width=10, edge_color='r', nodelist=array,
            node_color='r', ax=axr)

    st.pyplot(figr)
    st.write("预期检查到酒驾司机数", int(-x))

# 显示地图抽象图

# 设置大字体显示文字



# nx.draw(abstract_map, pos=dui_object.pos, ax=axr)
# nx.draw(abstract_map, pos=dui_object.pos, edgelist=[(0,1)], width=10, edge_color='r', nodelist=array, node_color='r', ax=axr)






