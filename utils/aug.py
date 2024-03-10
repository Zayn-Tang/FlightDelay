import copy
import numpy as np 
import torch 

def sim_global(flow_data, sim_type='cos'):
    """Calculate the global similarity of traffic flow data.
    :param flow_data: tensor, original flow [n,l,v,c] or location embedding [n,v,c]
    :param type: str, type of similarity, attention or cosine. ['att', 'cos']
    :return sim: tensor, symmetric similarity, [v,v]
    """
    if len(flow_data.shape) == 4:
        n,l,v,c = flow_data.shape
        att_scaling = n * l * c
        cos_scaling = torch.norm(flow_data, p=2, dim=(0, 1, 3)) ** -1 # cal 2-norm of each node, dim N
        sim = torch.einsum('btnc, btmc->nm', flow_data, flow_data)
    elif len(flow_data.shape) == 3:
        n,v,c = flow_data.shape
        att_scaling = n * c
        cos_scaling = torch.norm(flow_data, p=2, dim=(0, 2)) ** -1 # cal 2-norm of each node, dim N
        sim = torch.einsum('bnc, bmc->nm', flow_data, flow_data)
    else:
        raise ValueError('sim_global only support shape length in [3, 4] but got {}.'.format(len(flow_data.shape)))

    if sim_type == 'cos':
        # cosine similarity
        scaling = torch.einsum('i, j->ij', cos_scaling, cos_scaling)
        sim = sim * scaling
    elif sim_type == 'att':
        # scaled dot product similarity
        scaling = float(att_scaling) ** -0.5 
        sim = torch.softmax(sim * scaling, dim=-1)
    else:
        raise ValueError('sim_global only support sim_type in [att, cos].')
    
    return sim

def aug_topology(sim_mx, input_graph, percent=0.2):
    """Generate the data augumentation from topology (graph structure) perspective 
        for undirected graph without self-loop.
    :param sim_mx: tensor, symmetric similarity, [v,v]
    :param input_graph: tensor, adjacency matrix without self-loop, [v,v]
    :return aug_graph: tensor, augmented adjacency matrix on cuda, [v,v]
    """    
    ## edge dropping starts here
    drop_percent = percent / 2
    
    index_list = input_graph.nonzero() # list of edges [row_idx, col_idx]
    
    edge_num = int(index_list.shape[0] / 2)  # treat one undirected edge as two edges
    edge_mask = (input_graph > 0).tril(diagonal=-1)  # 对角线往下一个（包含）才会被认为是 True，主对角元素和上方元素都会设为False。 但是 input_graph 是对称的
    add_drop_num = int(edge_num * drop_percent / 2) 
    aug_graph = copy.deepcopy(input_graph) 

    sim_mx = sim_mx.to(input_graph.device)
    drop_prob = torch.softmax(sim_mx[edge_mask], dim=0)
    drop_prob = (1. - drop_prob).cpu().numpy() # normalized similarity to get sampling probability 
    drop_prob /= drop_prob.sum()
    drop_list = np.random.choice(edge_num, size=add_drop_num, p=drop_prob)
    drop_index = index_list[drop_list]
    
    zeros = torch.zeros_like(aug_graph[0, 0])
    aug_graph[drop_index[:, 0], drop_index[:, 1]] = zeros
    aug_graph[drop_index[:, 1], drop_index[:, 0]] = zeros

    ## edge adding starts here
    node_num = input_graph.shape[0]
    x, y = np.meshgrid(range(node_num), range(node_num), indexing='ij')
    mask = y < x
    x, y = x[mask], y[mask]

    add_prob = sim_mx[torch.ones(sim_mx.size(), dtype=bool).tril(diagonal=-1)] # .numpy()
    add_prob = torch.softmax(add_prob, dim=0).numpy()
    add_list = np.random.choice(int((node_num * node_num - node_num) / 2), 
                                size=add_drop_num, p=add_prob)
    
    ones = torch.ones_like(aug_graph[0, 0])
    aug_graph[x[add_list], y[add_list]] = ones
    aug_graph[y[add_list], x[add_list]] = ones
    
    return aug_graph

def aug_topology_(sim_mx, input_graph, percent=0.1):
    """Generate the data augumentation from topology (graph structure) perspective  
        for undirected graph without self-loop.
    :param sim_mx: tensor, symmetric similarity, [v,v]
    :param input_graph: tensor, adjacency matrix without self-loop, [v,v]
    :return aug_graph: tensor, augmented adjacency matrix on cuda, [v,v]
    """
    input_graph = input_graph.cpu().numpy()
    index_list = input_graph.nonzero() # list of edges [row_idx, col_idx]

    edge_num = index_list[0].size
    aug_num = int(edge_num * percent) 
    
    graph = input_graph[index_list]
    dist_graph = np.random.normal(graph.mean().item(), graph.std().item(), size=aug_num).astype(np.int32)
    dist_graph[dist_graph<0]=0

    edge_mask = input_graph>0
    # sim_mx = sim_mx.to(input_graph.device)
    drop_prob = torch.softmax(sim_mx[edge_mask], dim=0)
    drop_prob = (1. - drop_prob).cpu().numpy() # normalized similarity to get sampling probability 
    drop_prob /= drop_prob.sum()
    drop_list = np.random.choice(edge_num, size=aug_num, p=drop_prob)

    # index_list = np.stack([index_list[0], index_list[1]], axis=1)
    index_list = np.stack(index_list, axis=1)
    drop_index = index_list[drop_list]
    
    # 检查：不要改成0了，改成同分布采样
    input_graph[drop_index[:, 0], drop_index[:, 1]] = dist_graph

    input_graph = torch.from_numpy(input_graph)
    return input_graph


def aug_traffic(t_sim_mx, flow_data, percent=0.2):
    """Generate the data augumentation from traffic (node attribute) perspective.
    :param t_sim_mx: temporal similarity matrix after softmax, [l,n,v]
    :param flow_data: input flow data, [n,l,v,c]
    """
    l, n, v = t_sim_mx.shape
    mask_num = int(n * l * v * percent)
    aug_flow = copy.deepcopy(flow_data)

    mask_prob = (1. - t_sim_mx.permute(1, 0, 2).reshape(-1)).cpu().numpy()  # 越不相似越容易被 mask 掉
    mask_prob /= mask_prob.sum()

    x, y, z = np.meshgrid(range(n), range(l), range(v), indexing='ij')
    mask_list = np.random.choice(n * l * v, size=mask_num, p=mask_prob)

    zeros = torch.zeros_like(aug_flow[0, 0, 0])
    aug_flow[
        x.reshape(-1)[mask_list], 
        y.reshape(-1)[mask_list], 
        z.reshape(-1)[mask_list]] = zeros 

    return aug_flow
