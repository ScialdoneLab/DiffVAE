import numpy as np
import copy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

def gen_subgraph(adj_orig, edgeratio, sym):
    if sym:
        adj_subgraph = copy.deepcopy(adj_orig)
        upper_tri = adj_subgraph[np.triu_indices(adj_subgraph.shape[0])]
        nedges_to_remove = np.floor((1 - edgeratio) * upper_tri.sum()).astype(np.int64)
        edges = np.where(upper_tri == 1)[0]
        selected_edges = np.random.choice(edges, size = nedges_to_remove, replace = False)
        upper_tri[selected_edges] = 0
        adj_subgraph = np.zeros((adj_orig.shape[0], adj_orig.shape[0]))
        adj_subgraph[np.triu_indices(adj_orig.shape[0])] = upper_tri
        adj_subgraph = adj_subgraph + adj_subgraph.T - np.diag(np.diag(adj_subgraph))
    else:
        nedges_to_remove = np.floor((1 - edgeratio) * adj_orig.sum()).astype(np.int64)
        adj_subgraph = copy.deepcopy(adj_orig)
        edges = np.where(adj_subgraph.ravel() == 1)[0]
        selected_edges = np.random.choice(edges, size = nedges_to_remove, replace = False)
        adj_subgraph.ravel()[selected_edges] = 0

    return adj_subgraph

def gen_training_array(adj_orig, edgeratio_total, edgeratio_single, sym, examples):
    adj_training_total = gen_subgraph(adj_orig, edgeratio_total, sym)
    edgeratio_example = edgeratio_single / edgeratio_total
    training_array = gen_subgraph(adj_training_total, edgeratio_example, sym)
    for i in range(examples - 1):
        new_subgraph = gen_subgraph(adj_training_total, edgeratio_example, sym)
        training_array = np.array([training_array, new_subgraph])

    return training_array, adj_training_total

def preprocess_input_adj(adj_orig, sym, diag):
    if sym:
        adj_sym = adj_orig + adj_orig.T
        adj_sym[adj_sym > 1] = 1
    else:
        adj_sym = adj_orig
    if diag is not None:    
        np.fill_diagonal(adj_sym, diag)
    
    return adj_sym

def crop_isolated_nodes(adj_orig, feat_orig):
    adj_cropped = adj_orig
    feat_cropped = feat_orig
    
    node_degrees = np.sum(adj_orig, axis=1) + np.sum(adj_orig, axis=2)
    isolated_nodes = [i for i,d in enumerate(node_degrees) if d==0]
    np.delete(adj_cropped, isolated_nodes, axis=1)
    np.delete(adj_cropped, isolated_nodes, axis=2)
    np.delete(feat_cropped, isolated_nodes, axis=1)
    
    return(adj_cropped, feat_cropped)

def test_auc(adj_orig, adj_val, y_pred):
    adj_output = y_pred.flatten()[adj_val.flatten() == 0]
    adj_test = adj_orig.flatten()[adj_val.flatten() == 0]

    return roc_auc_score(y_true=adj_test, y_score=adj_output)

def test_classification_report(adj_orig, adj_val, y_pred):
    adj_output = y_pred.flatten()[adj_val.flatten() == 0]
    adj_output = adj_output > 0.5
    adj_test = adj_orig.flatten()[adj_val.flatten() == 0]

    return classification_report(y_true=adj_test, y_pred=adj_output)
