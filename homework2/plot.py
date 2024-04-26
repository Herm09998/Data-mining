import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Constructing sample data
frequent_itemsets = pd.DataFrame({
    'antecedents': [('N100',), ('N102',), ('N104', 'N105'), ('N107',)],
    'consequents': [('N101',), ('N103',), ('N106',), ('N108', 'N109')],
    'support': [0.015, 0.013, 0.012, 0.010],
    'confidence': [0.700, 0.650, 0.750, 0.500]
})

# Create the directed graph
G = nx.DiGraph()

# Add nodes and edges with attributes
for index, row in frequent_itemsets.iterrows():
    for ant in row['antecedents']:
        for cons in row['consequents']:
            G.add_node(ant, color='lightblue')
            G.add_node(cons, color='lightgreen')
            # Adding weights and support as edge attributes
            G.add_edge(ant, cons, weight=row['confidence']*10, support=row['support'])

# Node colors
node_color = [nx.get_node_attributes(G, 'color')[node] for node in G.nodes()]

# Edge weights
weights = [G[u][v]['weight'] for u, v in G.edges()]

# Position layout
pos = nx.spring_layout(G, seed=7)  # Set a random seed for layout consistency

# Draw the graph
plt.figure(figsize=(8, 8))
nx.draw(G, pos, node_color=node_color, width=weights, with_labels=True, font_size=10, node_size=2000)
edge_labels = nx.get_edge_attributes(G, 'support')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title('News Click Patterns')
plt.show()
