import numpy as np
from sklearn.metrics import roc_auc_score



def gen_subgraph(adj_orig, edgeratio, sim):
    if sim:
        nedges_to_remove = np.floor((1 - edgeratio) * sum(adj_orig) / 2)
        upper_tri = adj_orig[np.triu_indices(adj_orig.shape[0])]
        indices = np.where(upper_tri == 1)
        upper_tri[np.random.choice(indices, size = nedges_to_remove, replace = False)] = 0
        adj_subgraph = np.zeros((adj_orig.shape[0], adj_orig.shape[0]))
        adj_subgraph[np.triu_indices(adj_orig.shape[0])] = upper_tri
        adj_subgraph = adj_subgraph + adj_subgraph.T - np.diag(np.diag(X))
    else:
        nedges_to_remove = np.floor((1 - edgeratio) * adj_orig.sum()).astype(np.int64)
        edges = np.where(adj_orig.ravel() == 1)[0]
        adj_subgraph = adj_orig
        selected_edges = np.random.choice(edges, size = nedges_to_remove, replace = False)
        adj_subgraph.ravel()[selected_edges] = 0

    return adj_subgraph

# def val_eval(self, adj_orig, adj_output):
#
#     adj_output = adj_output.flat()[self.adj_train.flat() == 0]
#     adj_orig = adj_orig.flat()[self.adj_train.flat() == 0]
#
#     return roc_auc_score(y_true=adj_orig, y_score=adj_output)
