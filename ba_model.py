import networkx as nx
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import random

COLOR = '#40a6d1'

# Plotting

def k_distrib(graph, scale='lin', colour='#40a6d1', alpha=.8, fit_line=False, expct_lo=1, expct_hi=10, expct_const=1):

    plt.close()
    num_nodes = graph.number_of_nodes()
    max_degree = 0

    # Calculate the maximum degree to know the range of x-axis
    for n in graph.nodes():
        if graph.degree(n) > max_degree:
            max_degree = graph.degree(n)

    # X-axis and y-axis values
    x = []
    y_tmp = []

    # Loop over all degrees until the maximum to compute the portion of nodes for that degree
    for i in range(max_degree + 1):
        x.append(i)
        y_tmp.append(0)
        for n in graph.nodes():
            if graph.degree(n) == i:
                y_tmp[i] += 1
        y = [i / num_nodes for i in y_tmp]

    # Check for the lin / log parameter and set axes scale
    if scale == 'log':
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Degree distribution (log-log scale)')
        plt.ylabel('log(P(k))')
        plt.xlabel('log(k)')
        plt.plot(x, y, linewidth = 0, marker = 'o', markersize = 8, color = colour, alpha = alpha)

        if fit_line:
            # Add theoretical distribution line k^-3
            # Note that you need to parametrize it manually
            w = [a for a in range(expct_lo,expct_hi)]
            z = []
            for i in w:
                x = (i**-3) * expct_const # set line's length and fit intercept
                z.append(x)

            plt.plot(w, z, 'k-', color='#7f7f7f')

    else:
        plt.plot(x, y, linewidth = 0, marker = 'o', markersize = 8, color = colour, alpha = alpha)
        plt.title('Degree distribution (linear scale)')
        plt.ylabel('P(k)')
        plt.xlabel('k')

    plt.show()


# BA algo functions

def rand_prob_node():
    nodes_probs = []
    for node in G.nodes():
        node_degr = G.degree(node)
        #print(node_degr)
        node_proba = node_degr / (2 * len(G.edges()))
        #print("Node proba is: {}".format(node_proba))
        nodes_probs.append(node_proba)
        #print("Nodes probablities: {}".format(nodes_probs))
    random_proba_node = np.random.choice(G.nodes(),p=nodes_probs)
    #print("Randomly selected node is: {}".format(random_proba_node))
    return random_proba_node

def add_edge():
        if len(G.edges()) == 0:
            random_proba_node = 0
        else:
            random_proba_node = rand_prob_node()
        new_edge = (random_proba_node, new_node)
        if new_edge in G.edges():
            add_edge()
        else:
            G.add_edge(new_node, random_proba_node)
            # print("Edge added: {} {}".format(new_node + 1, random_proba_node))

if __name__ == "__main__":
    print("***\nWelcome to Barabási–Albert (BA) model simulation\n\n")

    # Get parameters
    init_nodes = int(input("Please type in the initial number of nodes (m_0): "))
    final_nodes = int(input("Please type in the final number of nodes: "))
    m_parameter = int(input("Please type in the value of m parameter (m<=m_0): "))
    remove_fraction = float(input("Fraction of the nodes to be removed: "))
    number_to_remove = int(remove_fraction * final_nodes)

    print("\n")
    print("Creating initial graph...")

    G = nx.complete_graph(init_nodes)

    print("Graph created. Number of nodes: {}".format(len(G.nodes())))
    print("Adding nodes...")

    count = 0
    new_node = init_nodes

    for f in range(final_nodes - init_nodes):
        # print("----------> Step {} <----------".format(count))
        G.add_node(init_nodes + count)
        # print("Node added: {}".format(init_nodes + count + 1))
        count += 1
        for e in range(0, m_parameter):
            add_edge()
        new_node += 1


    print("\nFinal number of nodes ({}) reached".format(len(G.nodes())))

    print("Draw the initial network")
    nx.draw(G, alpha = .3, edge_color = COLOR, node_color = COLOR, node_size=20)
    plt.savefig("init_topo_m0{}_n{}_m{}.png".format(init_nodes, final_nodes, m_parameter))
    plt.clf()

    print("max_diameter: {}".format(nx.diameter(G)))

    # start removing nodes
    print("* Number of node in topo: {}".format(G.number_of_nodes()))
    print("* Number of nodes to be removed: {}".format(number_to_remove))

    # remove nodes base
    print("---failure (remove nodes randomly)")
    random_rm_G = G.copy()
    RandomSample = random.sample(random_rm_G.nodes(), number_to_remove)
    random_rm_G.remove_nodes_from(RandomSample)

    sub_graphs = (random_rm_G.subgraph(c) for c in nx.connected_components(random_rm_G))
    # print("* Partitioned to {} sub graphs.".format(len(sub_graphs)))
    max_diameter = 0
    sub_graph_n = 0
    for i, sg in enumerate(sub_graphs):
        max_diameter = max(max_diameter, nx.diameter(sg))
        sub_graph_n += 1
    #    print("subgraph {} has {} nodes".format(i, sg.number_of_nodes()))
    print("max_diameter: {}, sub graph size: {}".format(max_diameter, sub_graph_n))

    nx.draw(random_rm_G, alpha=.3, edge_color=COLOR, node_color=COLOR, node_size=20)
    plt.savefig("removed_m0-{}_n{}_m{}_r{}.png".format(init_nodes, final_nodes, m_parameter, remove_fraction))
    plt.clf()

    # attack
    print("---Attack (remove node degree >= 5)")
    remove = [node for node, degree in dict(G.degree()).items() if degree >= 5]
    random.shuffle(remove)

    remove_number = number_to_remove if number_to_remove < len(remove) else len(remove)
    G.remove_nodes_from(remove[: remove_number])

    sub_graphs = (G.subgraph(c) for c in nx.connected_components(G))
    # print("* Partitioned to {} sub graphs.".format(len(sub_graphs)))
    max_diameter = 0
    sub_graph_n = 0
    for i, sg in enumerate(sub_graphs):
        max_diameter = max(max_diameter, nx.diameter(sg))
        sub_graph_n += 1
    #     print("subgraph {} has {} nodes".format(i, sg.number_of_nodes()))
    print("max_diameter: {}, sub graph size: {}".format(max_diameter, sub_graph_n))

    nx.draw(G, alpha=.3, edge_color=COLOR, node_color=COLOR, node_size=20)
    plt.savefig("attack_m0-{}_n{}_m{}_r{}.png".format(init_nodes, final_nodes, m_parameter, remove_fraction))
    plt.clf()
