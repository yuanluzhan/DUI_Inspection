import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dui_chiping import scenario
import matplotlib.pyplot as plt
import networkx as nx

# 创建一个标题
st.title('结果展示')

# 创建一个侧边栏
st.sidebar.header('输入信息')

# 在侧边栏中添加滑动条用于选择数据数量
num_polices = st.sidebar.slider('检查点数量', min_value=1, max_value=10, value=3, step=1)
num_drivers = st.sidebar.slider('司机数量', min_value=100, max_value=10000, value=500, step=100)
num_duidrivers = st.sidebar.slider('酒驾司机数', min_value=10, max_value=1000, value=50, step=10)
knowledge_strength = 0.3#st.sidebar.slider('Knowledge strength', min_value=0.0, max_value=1.0, value=0.3, step=0.05)
n_rows = st.sidebar.slider('行数', min_value=3, max_value=20, value=5, step=1)
n_columns = st.sidebar.slider('列数', min_value=3, max_value=20, value=5, step=1)
# Input array from the user
source_region = st.sidebar.text_input("输入酒驾司机经常出发的节点位置",value=1)


# 创建一个场景对象
dui_object = scenario(knowledge_strength,30,num_polices,num_drivers,num_duidrivers,1,n_rows,n_columns)






# 计算建议的策略
# x,y = dui_object.run()
#
# # 输出前k大的index
# def k_largest_indices(input_list, k):
#     return sorted(range(len(input_list)), key=lambda i: input_list[i], reverse=True)[:k]
#
# reco_index = k_largest_indices(x,int(num_polices))
# edges = list(dui_object.map.edges)
# reco_location = []
# for i in reco_index:
#     reco_location.append(edges[i])


# 检查输入是否合法


# 展示
abstract_map = dui_object.mapinit(n_rows,n_columns)


fig, ax = plt.subplots(figsize=(10,10))
array = [int(x.strip()) for x in source_region.split(',')]
# Display the input array
nx.draw(abstract_map,pos=dui_object.pos, ax=ax)
nx.draw(abstract_map,pos=dui_object.pos,nodelist=array, node_color='r',ax=ax)
st.write("地图抽象")
st.pyplot(fig)
st.write('<span style="font-size: 20px;">This is a larger text.</span>', unsafe_allow_html=True)
my_variable = "Hello, World!"
st.write(f"<span style='font-size: 20px; font-weight: bold;'>{my_variable}</span>", unsafe_allow_html=True)
# st.write("预期检查到酒驾司机数", int(y))
# figr, axr = plt.subplots(figsize=(20, 20))
# nx.draw(abstract_map, pos=dui_object.pos, ax=axr)
# # nx.draw(abstract_map, pos=dui_object.pos, nodelist=array, node_color='r', ax=ax)
# nx.draw(abstract_map, pos=dui_object.pos, edgelist=reco_location,width = 10, edge_color='r',nodelist=array, node_color='r',ax=axr)
# st.write("推荐检查点设置")
# st.pyplot(figr)






