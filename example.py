import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


# karate = nx.karate_club_graph()
# nx.draw(karate, pos=nx.circular_layout(karate))
# plt.draw()
# plt.show()
# x2 = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
#                [[10, 11, 12],[13, 14, 15], [16, 17, 18]]])
# print(x2[1:, :])
# unchanged_nodes_list = [{1}, {3}, {5}]
# result_array = np.zeros(3)
# zero = np.zeros(3)
# A = np.array([[1, 3, 5], [2, 4, 6],[3, 5, 7]])
# result_array = np.vstack([result_array, A])
# print(result_array)
# print(result_array[1:])
# print(zero + result_array)
# B = '0'
# print(B[0])


G = nx.random_regular_graph(3, 20)
G1 = nx.barabasi_albert_graph(20, 3)
G2 = nx.watts_strogatz_graph(20, 4, 0)
G3 = nx.complete_graph(20)
G4 = nx.erdos_renyi_graph(20, 0.15)
nx.draw_circular(G2, with_labels=False, node_color='#0165fc', edge_color='#0165fc')
# nx.draw_circular(G1, with_labels=False)
# nx.draw_circular(G2, with_labels=False)
# nx.draw_circular(G3, with_labels=False)
# nx.draw_circular(G4, with_labels=False)

plt.show()