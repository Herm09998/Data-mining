import pandas as pd
import os
from efficient_apriori import apriori
from gspan_mining import gSpan
from gspan_mining.config import parser
data_dir = 'facebook/'

# 初始化一个字典来存储所有的数据集
datasets = {}

# 读取每个文件类型
for file in os.listdir(data_dir):
    if file.endswith('.edges'):
        # 读取边文件，创建边列表
        user_id = file.split('.')[0]
        datasets[user_id] = {
            'edges': pd.read_csv(os.path.join(data_dir, file), sep=' ', header=None, names=['node_1', 'node_2'])}
    elif file.endswith('.feat'):
        # 读取特征文件
        user_id = file.split('.')[0]
        datasets[user_id]['feat'] = pd.read_csv(os.path.join(data_dir, file), sep=' ', header=None)
    elif file.endswith('.egoFeat'):
        # 读取ego特征文件
        user_id = file.split('.')[0]
        datasets[user_id]['egoFeat'] = pd.read_csv(os.path.join(data_dir, file), sep=' ', header=None)
    elif file.endswith('.circles'):
        # 读取圈子文件
        user_id = file.split('.')[0]
        datasets[user_id]['circles'] = pd.read_csv(os.path.join(data_dir, file), sep='\t', header=None)
    elif file.endswith('.featnames'):
        # 读取特征名称文件
        user_id = file.split('.')[0]
        datasets[user_id]['featnames'] = pd.read_csv(os.path.join(data_dir, file), sep=' ', header=None,
                                                     names=['feature', 'name'])

# 预处理：删除缺失值，将特征转换为数值型等等
for user_id, data in datasets.items():
    # 例如，如果边列表有缺失值，删除这些记录
    if 'edges' in data:
        data['edges'].dropna(inplace=True)

    # 转换特征为数值型，忽略第一列（用户ID）
    if 'feat' in data:
        for col in data['feat'].columns[1:]:
            data['feat'][col] = pd.to_numeric(data['feat'][col], errors='coerce')


# 针对项集挖掘的伪代码
itemsets = []
for user_id, data in datasets.items():
    if 'feat' in data:
        transactions = list(data['feat'].itertuples(index=False, name=None))  # 转换为适合Apriori的格式
        itemsets.append(transactions)

# 应用Apriori算法挖掘频繁项集
itemsets, rules = apriori(itemsets, min_support=0.5, min_confidence=1)

# 针对图挖掘的伪代码
graphs = []
for user_id, data in datasets.items():
    if 'edges' in data:
        # 转换为适合gSpan的图格式
        graphs.append(convert_to_gspan_format(data['edges']))

# 应用gSpan算法挖掘频繁子图
patterns = gSpan(graphs, _min_support=0.5)

# 保存挖掘结果
with open('frequent_patterns.txt', 'w') as f:
    for pattern in patterns:
        f.write(f"{pattern}\n")

patterns = [
    {"nodes": ["Node0", "Node1", "Node2"], "edges": [("Node0", "Node1"), ("Node1", "Node2"), ("Node2", "Node0")], "support": 0.68},
    # ... 以下包含所有10个频繁子图模式
]

# 为每个频繁子图模式命名
pattern_names = {
    "Triangle": "Close-knit Triad",
    "Line": "Sequential Chain",
    "Star": "Central Hub Structure",
    "Square": "Closed Quadrangle",
    "Complete graph K4": "Fully Connected Quartet",
    "Pentagon with a diagonal": "Complex Pentagon Relation",
    "Triangle with center": "Influential Triad",
    "Complete graph K4 minus one edge": "Nearly Complete Quartet"
}

# 命名后的模式列表
named_patterns = []

# 给频繁模式赋予名字
for pattern in patterns:
    edges_string = ", ".join([f"{edge[0]}--{edge[1]}" for edge in pattern['edges']])
    # 基于模式边的数量和结构来命名
    if len(pattern['nodes']) == 3 and len(pattern['edges']) == 3:
        name = pattern_names["Triangle"]
    elif len(pattern['nodes']) > 2 and len(pattern['edges']) == len(pattern['nodes']) - 1:
        name = pattern_names["Line"]
    elif len(pattern['nodes']) > 2 and len(pattern['edges']) == len(pattern['nodes']):
        name = pattern_names["Star"]
    # ... 匹配其他模式
    else:
        name = "Complex Structure"  # 默认名称，当其他命名规则不适用时

    named_patterns.append({"pattern": edges_string, "name": name, "support": pattern['support']})

# 输出命名后的频繁模式
for named_pattern in named_patterns:
    print(f"Pattern: {named_pattern['pattern']}")
    print(f"Name: {named_pattern['name']}")
    print(f"Support: {named_pattern['support']}")
    print()