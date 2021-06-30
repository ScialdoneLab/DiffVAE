"""
Code Author: Ioana Bica (ioana.bica95@gmail.com), Marco Stock (marco.stock@tum.de)
"""

import os
import argparse
import numpy as np
import pandas as pd

from autoencoder_models.GraphDiffVAE import GraphDiffVAE
from data.data_processing import get_gene_expression_data
from data.data_processing import scale_gene_expression_df
from data.build_graphs import build_correlation_graph

from GRN_functions import gen_subgraph

import time

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene_expression_filename", default='data/Zebrafish/GE_mvg.csv')
    parser.add_argument("--hidden_dimensions", default=[512], nargs="*", type=int)
    parser.add_argument("--latent_dimension", default=50, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--model_name", default='graph_test')
    parser.add_argument("--input_adj_matrix")
    parser.add_argument("--initial_node_features")
    parser.add_argument("--loss", default="categorical") #categorical, binary or f1
    parser.add_argument("--model", default="best") #best or last model
    parser.add_argument("--edgeratio", default=0.5, type=float) #ratio of edges used for training
    parser.add_argument("--kl_weight", default=1, type=float) #weight of KL loss in total sum


    return parser.parse_args()

if __name__ == '__main__':

    args = init_arg()
    if not os.path.exists('results/Graphs'):
        os.mkdir('results/Graphs')
    if not os.path.exists('results/Models'):
        os.mkdir('results/Models')

    model_timestamp = time.strftime("%Y%m%d_%H%M%S")

    if args.input_adj_matrix is None:
        gene_expression_normalized = get_gene_expression_data(args.gene_expression_filename)
        adj_matrix, initial_node_features = build_correlation_graph(gene_expression_normalized, num_neighbors=2)
    else:
        adj_matrix = np.genfromtxt(args.input_adj_matrix, delimiter=';', dtype='float64')
        initial_node_features_raw =  np.genfromtxt(args.initial_node_features, delimiter=';', dtype='float64')
        initial_node_features = scale_gene_expression_df(initial_node_features_raw)

    print("Dimensions of initial_node_features: " + str(initial_node_features.shape[0]) + ":"+ str(initial_node_features.shape[1]))
    print("Dimensions of adj_matrix: " + str(adj_matrix.shape[0]) + ":"+ str(adj_matrix.shape[1]))
    #np.savetxt('results/Graphs/' + model_timestamp + '_input_adj_matrix_' + args.model_name + '.csv', adj_matrix, delimiter=";")
    #np.savetxt('results/Graphs/' + model_timestamp + '_input_node_feat_' + args.model_name + '.csv', initial_node_features, delimiter=";")

    GraphVAE_model=GraphDiffVAE(num_nodes=adj_matrix.shape[0], num_features=initial_node_features.shape[1],
                                adj_matrix=adj_matrix, latent_dim=args.latent_dimension,
                                hidden_layers_dim=args.hidden_dimensions,
                                epochs=args.epochs,
                                learning_rate=args.learning_rate,
                                loss_mode=args.loss,
                                model_select=args.model,
                                kl_weight=args.kl_weight,
                                timestamp=model_timestamp)

    adj_train = gen_subgraph(adj_matrix, edgeratio=args.edgeratio, sim=False)
    np.savetxt('results/Graphs/' + model_timestamp + '_subgraph_' + str(int(100*args.edgeratio)) + '_' + args.model_name + '.csv', adj_matrix, delimiter=";")
    predictions, latent_res = GraphVAE_model.train_vae(initial_node_features, adj_matrix, adj_train)
    np.savetxt('results/Graphs/' + model_timestamp + '_pred_adj_' + args.model_name + '_hidden_' + "_".join(str(n) for n in args.hidden_dimensions) + '_latent_' + str(args.latent_dimension) + '_' + args.loss +'.csv', predictions, delimiter=";")
    #np.savetxt('results/Graphs/' + model_timestamp + '_node_latent_' + args.model_name + '.csv', latent_res, delimiter=";")
