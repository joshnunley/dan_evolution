import torch
import networkx as nx

def get_abstract_graph(type, device):
    abstract_graph_nodes = []
    abstract_graph_edges = {}

    grid_size = (28, 28)
    directed_graph = nx.grid_2d_graph(grid_size[0], grid_size[1]).to_directed()

    for node in directed_graph.nodes:
        directed_graph.add_edge(node, node)
        
        if directed_graph.has_node((node[0] + 1, node[1] + 1)):
            directed_graph.add_edge(node, (node[0] + 1, node[1] + 1))
        if directed_graph.has_node((node[0] + 1, node[1] - 1)):
            directed_graph.add_edge(node, (node[0] + 1, node[1] - 1))
        if directed_graph.has_node((node[0] - 1, node[1] + 1)):
            directed_graph.add_edge(node, (node[0] - 1, node[1] + 1))
        if directed_graph.has_node((node[0] - 1, node[1] - 1)):
            directed_graph.add_edge(node, (node[0] - 1, node[1] - 1))

    def graph_to_adjacency(directed_graph, device):
        # create a numpy array of zeros with the same shape as the adjacency matrix
        # and then fill in the ones
        adjacency_matrix = nx.to_numpy_array(directed_graph)
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32, device=device)
        return adjacency_matrix

    grid_adjacency = graph_to_adjacency(directed_graph, device)

    output_mask = torch.zeros(1, 28, 28, device=device)
    output_mask[:, 12:16, 12:16] = 1
    output_mask = output_mask.reshape(1, 28*28)

    if type == 'convolution':
        abstract_graph_nodes = [
            {
                "num_neurons": 784,
                "input_size": 1,
                "output_indices": [],
                "state_update_id": 0,
                "query_id": 0,
                "adjacency_masks_provided": False,
            },
            {
                "num_neurons": 784,
                "input_size": 0,
                "output_indices": [],
                "state_update_id": 2,
                "query_id": 1,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 784,
                "input_size": 0,
                "output_indices": [],
                "state_update_id": 3,
                "query_id": 1,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 784,
                "input_size": 0,
                "output_indices": [],
                "state_update_id": 4,
                "query_id": 1,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 784,
                "input_size": 0,
                "output_indices": [],
                "state_update_id": 5,
                "query_id": 2,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 10,
                "input_size": 0,
                "output_indices": [0],
                "state_update_id": 1,
                "query_id": 0,
                "adjacency_masks_provided": False,
            },
        ]
        abstract_graph_edges = {
            (0, 1): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (1, 1): {"key_id": 0, "adjacency_mask": torch.eye(28*28, device=device)},
            (0, 2): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (2, 2): {"key_id": 0, "adjacency_mask": torch.eye(28*28, device=device)},
            (0, 3): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (3, 3): {"key_id": 0, "adjacency_mask": torch.eye(28*28, device=device)},
            (1, 4): {"key_id": 0, "adjacency_mask": torch.eye(784, device=device)},
            (2, 4): {"key_id": 0, "adjacency_mask": torch.eye(784, device=device)},
            (3, 4): {"key_id": 0, "adjacency_mask": torch.eye(784, device=device)},
            (4, 4): {"key_id": 0, "adjacency_mask": torch.eye(784, device=device)},
            (4, 5): {"key_id": 0},
            (5, 5): {"key_id": 0},
        }
    elif type == 'basic':
        abstract_graph_nodes = [
            {
                "num_neurons": 784,
                "input_size": 1,
                "output_indices": [],
                "state_update_id": 0,
                "query_id": 0,
                "adjacency_masks_provided": False,
            },
            {
                "num_neurons": 10,
                "input_size": 0,
                "output_indices": [0],
                "state_update_id": 1,
                "query_id": 0,
                "adjacency_masks_provided": False,
            },
        ]
        abstract_graph_edges = {
            (0, 1): {"key_id": 0},
            (1, 1): {"key_id": 0},
        }
    elif type == 'basic_fully_connected_input':
        abstract_graph_nodes = [
            {
                "num_neurons": 784,
                "input_size": 1,
                "output_indices": [],
                "state_update_id": 0,
                "query_id": 0,
                "adjacency_masks_provided": False,
            },
            {
                "num_neurons": 10,
                "input_size": 0,
                "output_indices": [0],
                "state_update_id": 1,
                "query_id": 1,
                "adjacency_masks_provided": False,
            },
        ]
        abstract_graph_edges = {
            (0, 0): {"key_id": 0},
            (0, 1): {"key_id": 0},
            (1, 1): {"key_id": 0},
        }
    elif type == 'basic_grid_connected_input':
        abstract_graph_nodes = [
            {
                "num_neurons": 784,
                "input_size": 1,
                "output_indices": [],
                "state_update_id": 0,
                "query_id": 0,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 10,
                "input_size": 0,
                "output_indices": [0],
                "state_update_id": 1,
                "query_id": 1,
                "adjacency_masks_provided": False,
            },
        ]
        abstract_graph_edges = {
            (0, 0): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (0, 1): {"key_id": 0},
            (1, 1): {"key_id": 0},
        }
    elif type == 'two_layer_receptive_field':
        abstract_graph_nodes = [
            {
                "num_neurons": 784,
                "input_size": 1,
                "output_indices": [],
                "state_update_id": 0,
                "query_id": 0,
                "adjacency_masks_provided": False,
            },
            {
                "num_neurons": 784,
                "input_size": 0,
                "output_indices": [],
                "state_update_id": 1,
                "query_id": 1,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 10,
                "input_size": 0,
                "output_indices": [0],
                "state_update_id": 2,
                "query_id": 2,
                "adjacency_masks_provided": False,
            },
        ]
        abstract_graph_edges = {
            (0, 0): {"key_id": 0},
            (0, 1): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (1, 1): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (1, 2): {"key_id": 0},
            (2, 2): {"key_id": 0},
        }
    # NOTE: This seems to be unstable at a larger number of evals
    # but it appears that the recurrent connection below
    # stabilizes it
    # I got 97.17% accuracy with hidden state size 10 and 
    # n_evals=15 with this model. It doesn't look like the
    # recurrent version perform quite as well, but they may be more stable.
    # They also learn more quickly, though they stop learning more quickly as well.

    # NOTE: That's mostly wrong. Recurrent connections don't add stability, but
    # they do make the network learn faster and stop learning faster. Increase stability
    # using gradient accumulation. And maybe by decreasing gamma.
    elif type == 'three_layer_receptive_field':
        abstract_graph_nodes = [
            {
                "num_neurons": 784,
                "input_size": 1,
                "output_indices": [],
                "state_update_id": 0,
                "query_id": 0,
                "adjacency_masks_provided": False,
            },
            {
                "num_neurons": 784,
                "input_size": 0,
                "output_indices": [],
                "state_update_id": 1,
                "query_id": 1,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 784,
                "input_size": 0,
                "output_indices": [],
                "state_update_id": 2,
                "query_id": 2,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 10,
                "input_size": 0,
                "output_indices": [0],
                "state_update_id": 3,
                "query_id": 3,
                "adjacency_masks_provided": False,
            },
        ]
        abstract_graph_edges = {
            (0, 0): {"key_id": 0},
            (0, 1): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (1, 1): {"key_id": 0, "adjacency_mask": grid_adjacency},            
            (1, 2): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (2, 2): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (2, 3): {"key_id": 0},
            (3, 3): {"key_id": 0},
        }
    elif type == 'four_layer_receptive_field':
        abstract_graph_nodes = [
            {
                "num_neurons": 784,
                "input_size": 1,
                "output_indices": [],
                "state_update_id": 0,
                "query_id": 0,
                "adjacency_masks_provided": False,
            },
            {
                "num_neurons": 784,
                "input_size": 0,
                "output_indices": [],
                "state_update_id": 1,
                "query_id": 1,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 784,
                "input_size": 0,
                "output_indices": [],
                "state_update_id": 2,
                "query_id": 2,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 784,
                "input_size": 0,
                "output_indices": [],
                "state_update_id": 3,
                "query_id": 3,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 10,
                "input_size": 0,
                "output_indices": [0],
                "state_update_id": 4,
                "query_id": 4,
                "adjacency_masks_provided": False,
            },
        ]
        abstract_graph_edges = {
            (0, 0): {"key_id": 0},
            (0, 1): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (1, 1): {"key_id": 0, "adjacency_mask": grid_adjacency},            
            (1, 2): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (2, 2): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (2, 3): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (3, 3): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (3, 4): {"key_id": 0},
            (4, 4): {"key_id": 0},
        }
    elif type == 'three_layer_receptive_field_recurrent':
        abstract_graph_nodes = [
            {
                "num_neurons": 784,
                "input_size": 1,
                "output_indices": [],
                "state_update_id": 0,
                "query_id": 0,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 784,
                "input_size": 0,
                "output_indices": [],
                "state_update_id": 1,
                "query_id": 1,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 784,
                "input_size": 0,
                "output_indices": [],
                "state_update_id": 2,
                "query_id": 2,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 10,
                "input_size": 0,
                "output_indices": [0],
                "state_update_id": 3,
                "query_id": 3,
                "adjacency_masks_provided": False,
            },
        ]
        abstract_graph_edges = {
            (0, 0): {"key_id": 0, "adjacency_mask": torch.ones(784, 784, device=device)},
            (2, 0): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (0, 1): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (1, 1): {"key_id": 0, "adjacency_mask": grid_adjacency},            
            (1, 2): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (2, 2): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (2, 3): {"key_id": 0},
            (3, 3): {"key_id": 0},
        }
    elif type == 'four_layer_receptive_field_recurrent':
        abstract_graph_nodes = [
            {
                "num_neurons": 784,
                "input_size": 1,
                "output_indices": [],
                "state_update_id": 0,
                "query_id": 0,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 784,
                "input_size": 0,
                "output_indices": [],
                "state_update_id": 1,
                "query_id": 1,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 784,
                "input_size": 0,
                "output_indices": [],
                "state_update_id": 2,
                "query_id": 2,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 784,
                "input_size": 0,
                "output_indices": [],
                "state_update_id": 3,
                "query_id": 3,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 10,
                "input_size": 0,
                "output_indices": [0],
                "state_update_id": 4,
                "query_id": 4,
                "adjacency_masks_provided": False,
            },
        ]
        abstract_graph_edges = {
            (0, 0): {"key_id": 0, "adjacency_mask": torch.ones(784, 784, device=device)},
            (3, 0): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (0, 1): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (1, 1): {"key_id": 0, "adjacency_mask": grid_adjacency},            
            (1, 2): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (2, 2): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (2, 3): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (3, 3): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (3, 4): {"key_id": 0},
            (4, 4): {"key_id": 0},
        }
    elif type == 'two_layer_basic_recurrent':
        abstract_graph_nodes = [
            {
                "num_neurons": 784,
                "input_size": 1,
                "output_indices": [],
                "state_update_id": 0,
                "query_id": 0,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 784,
                "input_size": 0,
                "output_indices": [],
                "state_update_id": 1,
                "query_id": 1,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 10,
                "input_size": 0,
                "output_indices": [0],
                "state_update_id": 2,
                "query_id": 2,
                "adjacency_masks_provided": False,
            },
        ]
        abstract_graph_edges = {
            (0, 0): {"key_id": 0, "adjacency_mask": torch.eye(784, device=device)},
            (1, 0): {"key_id": 0, "adjacency_mask": torch.eye(784, device=device)},
            (0, 1): {"key_id": 0, "adjacency_mask": torch.eye(784, device=device)},
            (1, 1): {"key_id": 0, "adjacency_mask": torch.eye(784, device=device)},
            (1, 2): {"key_id": 0},
            (2, 2): {"key_id": 0},
        }
    elif type == 'two_layer_basic':
        abstract_graph_nodes = [
            {
                "num_neurons": 784,
                "input_size": 1,
                "output_indices": [],
                "state_update_id": 0,
                "query_id": 0,
                "adjacency_masks_provided": False,
            },
            {
                "num_neurons": 784,
                "input_size": 0,
                "output_indices": [],
                "state_update_id": 1,
                "query_id": 1,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 10,
                "input_size": 0,
                "output_indices": [0],
                "state_update_id": 2,
                "query_id": 2,
                "adjacency_masks_provided": False,
            },
        ]
        abstract_graph_edges = {
            (0, 1): {"key_id": 0, "adjacency_mask": torch.eye(784, device=device)},
            (1, 1): {"key_id": 0, "adjacency_mask": torch.eye(784, device=device)},
            (1, 2): {"key_id": 0},
            (2, 2): {"key_id": 0},
        }
    elif type == 'two_layer_basic_grid_grid_self_connected':
        abstract_graph_nodes = [
            {
                "num_neurons": 784,
                "input_size": 1,
                "output_indices": [],
                "state_update_id": 0,
                "query_id": 0,
                "adjacency_masks_provided": False,
            },
            {
                "num_neurons": 784,
                "input_size": 0,
                "output_indices": [],
                "state_update_id": 1,
                "query_id": 1,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 10,
                "input_size": 0,
                "output_indices": [0],
                "state_update_id": 2,
                "query_id": 2,
                "adjacency_masks_provided": False,
            },
        ]
        abstract_graph_edges = {
            (0, 0): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (0, 1): {"key_id": 0, "adjacency_mask": torch.eye(784, device=device)},
            (1, 1): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (1, 2): {"key_id": 0},
            (2, 2): {"key_id": 0},
        }
    elif type == 'two_layer_basic_fully_grid_self_connected':
        abstract_graph_nodes = [
            {
                "num_neurons": 784,
                "input_size": 1,
                "output_indices": [],
                "state_update_id": 0,
                "query_id": 0,
                "adjacency_masks_provided": False,
            },
            {
                "num_neurons": 784,
                "input_size": 0,
                "output_indices": [],
                "state_update_id": 1,
                "query_id": 1,
                "adjacency_masks_provided": True,
            },
            {
                "num_neurons": 10,
                "input_size": 0,
                "output_indices": [0],
                "state_update_id": 2,
                "query_id": 2,
                "adjacency_masks_provided": False,
            },
        ]
        abstract_graph_edges = {
            (0, 0): {"key_id": 0},
            (0, 1): {"key_id": 0, "adjacency_mask": torch.eye(784, device=device)},
            (1, 1): {"key_id": 0, "adjacency_mask": grid_adjacency},
            (1, 2): {"key_id": 0},
            (2, 2): {"key_id": 0},
        }
        
        
    output_id = len(abstract_graph_nodes) - 1

    return abstract_graph_nodes, abstract_graph_edges, output_id

