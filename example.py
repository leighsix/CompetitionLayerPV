import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# x2 = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
#                [[10, 11, 12],[13, 14, 15], [16, 17, 18]]])
# print(x2[1:, :])
unchanged_nodes_list = [{1}, {3}, {5}]

for i, nodes_list in enumerate(unchanged_nodes_list):
    print(i, nodes_list)

# G = nx.random_regular_graph(4, 20)
# G1 = nx.barabasi_albert_graph(20, 4)
# G2 = nx.watts_strogatz_graph(20, 4, 0.6)
# G3 = nx.complete_graph(20)
# G4 = nx.erdos_renyi_graph(20, 0.2)
# nx.draw_circular(G4, with_labels=False, node_color='#A0CBE2', edge_color='#A0CBE2')
# # nx.draw_circular(G1, with_labels=False)
# # nx.draw_circular(G2, with_labels=False)
# # nx.draw_circular(G3, with_labels=False)
# # nx.draw_circular(G4, with_labels=False)

# plt.show()