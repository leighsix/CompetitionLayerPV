from pymnet import *
import matplotlib.pyplot as plt
import SettingSimulationValue
import InterconnectedLayerModeling
import OpinionDynamics
import DecisionDynamics
from mpl_toolkits.mplot3d.axes3d import *
import matplotlib.animation as animation
import matplotlib
import numpy as np
from matplotlib import animation, rc
matplotlib.use("TkAgg")


class InterconnectedNetworkVisualization:
    @staticmethod
    def making_competition_movie(setting, inter_layer, p, v, save_files, using_prob, unchanged_nodes):
        fig = plt.figure()
        ims = []
        im = InterconnectedNetworkVisualization.draw_interconnected_network(setting, inter_layer, save_files)
        ims.append([im])
        for step_number in range(setting.Limited_step+1):
            state = []
            for i in inter_layer.A_nodes + inter_layer.B_nodes:
                state.append(inter_layer.two_layer_graph.nodes[i]['state'])
            state = np.array(state)
            if (np.all(state < 0) == 1) or (np.all(state > 0) == 1):
                OpinionDynamics.OpinionDynamics(setting, inter_layer, p, v, 0, using_prob, unchanged_nodes)
                im = InterconnectedNetworkVisualization.draw_interconnected_network(setting, inter_layer, save_files)
                ims.append([im])
                DecisionDynamics.DecisionDynamics(setting, inter_layer, v, 0, unchanged_nodes)
                im = InterconnectedNetworkVisualization.draw_interconnected_network(setting, inter_layer, save_files)
                ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        ani.save('dynamic_images.mp4')
        plt.show()


    @staticmethod
    def making_layer_A_graph(inter_layer, interconnected_network):
        interconnected_network.add_layer('layer A')
        for i in inter_layer.A_nodes:
            interconnected_network.add_node(i)
        for i, j in inter_layer.edges_on_A:
            interconnected_network[i, j, 'layer A'] = 1
        return interconnected_network

    @staticmethod
    def making_layer_B_graph(inter_layer, interconnected_network):
        interconnected_network.add_layer('layer B')
        for i in inter_layer.B_nodes:
            interconnected_network.add_node(i)
        for i, j in inter_layer.edges_on_B:
            interconnected_network[i, j, 'layer B'] = 1
        return interconnected_network

    @staticmethod
    def making_interconnected_layer(inter_layer):
        interconnected_network = MultilayerNetwork(aspects=1)
        InterconnectedNetworkVisualization.making_layer_A_graph(inter_layer, interconnected_network)
        InterconnectedNetworkVisualization.making_layer_B_graph(inter_layer, interconnected_network)
        for i, j in sorted(inter_layer.edges_on_AB):
            interconnected_network[i, 'layer A'][j, 'layer B'] = 1
        return interconnected_network

    @staticmethod
    def making_node_color(setting, inter_layer):
        node_color_dic = {}
        for i in inter_layer.A_nodes:
            node_color_dic[(i, 'layer A')] = setting.NodeColorDict[inter_layer.two_layer_graph.nodes[i]['state']]
        for i in inter_layer.B_nodes:
            node_color_dic[(i, 'layer B')] = setting.NodeColorDict[inter_layer.two_layer_graph.nodes[i]['state']]
        return node_color_dic

    @staticmethod
    def making_edge_color(setting, inter_layer):
        edge_color_dic = {}
        for i, j in inter_layer.edges_on_A:
            a = inter_layer.two_layer_graph.nodes[i]['state']
            b = inter_layer.two_layer_graph.nodes[j]['state']
            edge_color_dic[(i, 'layer A'), (j, 'layer A')] = setting.EdgeColorDict[a * b]
        for i, j in inter_layer.edges_on_B:
            a = inter_layer.two_layer_graph.nodes[i]['state']
            b = inter_layer.two_layer_graph.nodes[j]['state']
            edge_color_dic[(i, 'layer B'), (j, 'layer B')] = setting.EdgeColorDict[a * b]
        for i, j in inter_layer.edges_on_AB:
            a = inter_layer.two_layer_graph.nodes[j]['state']
            b = inter_layer.two_layer_graph.nodes[i]['state']
            edge_color_dic[(i, 'layer A'), (j, 'layer B')] = setting.EdgeColorDict[a * b]
        return edge_color_dic

    @staticmethod
    def draw_interconnected_network(setting, inter_layer, save_file_name):
        ax = plt.axes(projection='3d')
        draw(InterconnectedNetworkVisualization.making_interconnected_layer(inter_layer), layout='circular',
             layergap=1.3,
             layershape='rectangle',
             nodeCoords=inter_layer.node_location,
             nodelayerCoords={},
             layerPadding=0.03, alignedNodes=True, ax=ax, layerColorDict={'layer A': 'pink', 'layer B': 'steelblue'},
             layerColorRule={},
             edgeColorDict=InterconnectedNetworkVisualization.making_edge_color(setting, inter_layer),
             edgeColorRule={},
             edgeWidthDict={}, edgeWidthRule={}, defaultEdgeWidth=0.4, edgeStyleDict={},
             edgeStyleRule={'rule': 'edgetype', 'inter': ':', 'intra': '-'}, defaultEdgeStyle='-',
             nodeLabelDict={}, nodeLabelRule={}, defaultNodeLabel=None,
             nodeColorDict=InterconnectedNetworkVisualization.making_node_color(setting, inter_layer), nodeColorRule={},
             defaultNodeColor=None,
             nodeLabelColorDict={}, nodeLabelColorRule={}, defaultNodeLabelColor='k',
             nodeSizeDict={}, nodeSizeRule={"propscale": 0.05, 'rule': 'degree'}, defaultNodeSize=None)
        plt.savefig(save_file_name)
        im = plt.imshow(plt.imread(save_file_name), animated=True)
        return im



if __name__ == "__main__":
    print("Interconnected Layer Modeling")
    setting = SettingSimulationValue.SettingSimulationValue()
    inter_layer = InterconnectedLayerModeling.InterconnectedLayerModeling(setting)
    # InterconnectedNetworkVisualization.draw_interconnected_network(setting, inter_layer, 'result.png')
    InterconnectedNetworkVisualization.making_competition_movie(setting, inter_layer, 0.1, 0.1, 'result.png',
                                                                using_prob=None, unchanged_nodes=None)
    plt.show()
    print("Operating finished")
