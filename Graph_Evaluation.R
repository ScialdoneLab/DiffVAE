datetime_of_data = "20210709_135017" 

training_graph <- read.table(paste("./results/Graphs/", datetime_of_data, "_training_subgraph_Yeast_Gasch.csv", sep=""), header=FALSE, sep=";")
validation_graph <- read.table(paste("./results/Graphs/", datetime_of_data, "_validation_subgraph_Yeast_Gasch.csv", sep=""), header=FALSE, sep=";")
output_graph <- read.table(paste("./results/Graphs/", datetime_of_data, "_pred_adj_Yeast_Gasch_hidden_45_latent_4_categorical.csv", sep=""), header=FALSE, sep=";")
orig_graph_raw <- read.table("./data/Yeast/adj_matrix_yeast_gasch.csv", header=FALSE, sep=";")
orig_graph <- build_sym_adj(orig_graph_raw)

hist(unlist(output_graph))
#sum(sum(training_graph))
#hist(unlist(output_graph[output_graph>0.75]))
#sum(is.na(output_graph))

grn_eval <- function(orig_graph, training_graph, output_graph, mode, threshold) {
  if(mode=="test_asymetric") {
    print("------Evaluation of Validation/Test Set performance-------")
    tp <- sum(orig_graph >= 0.5 & training_graph < 0.5 & output_graph >= threshold)
    tn <- sum(orig_graph < 0.5 & training_graph < 0.5 & output_graph < threshold)
    fp <- sum(orig_graph < 0.5 & training_graph < 0.5 & output_graph >= threshold)
    fn <- sum(orig_graph >= 0.5 & training_graph < 0.5 & output_graph < threshold)
  }
  else if (mode=="train_asymetric") {
    print("------Evaluation of Training Set performance-------")
    tp <- sum(training_graph >= 0.5 & output_graph >= threshold)
    tn <- sum(training_graph < 0.5 & output_graph < threshold)
    fp <- sum(training_graph < 0.5 & output_graph >= threshold)
    fn <- sum(training_graph >= 0.5 & output_graph < threshold)
  }
  print_eval(tp, tn, fp, fn)
}

print_eval <- function(tp, tn, fp, fn) {
  #Precision
  ppv <- tp / (tp+fp)
  #Sensitivity
  tpr <- tp / (tp+fn)
  #Specificity
  tnr <- tn / (tn+fp)
  #Accuracy
  acc <- (tp+tn) / (tp+tn+fp+fn)
  #F1-score
  f1 <- 2*tp / (2*tp+fp+fn)
  #DOR
  dor <- tp*tn/(fp*fn)
  #FPR
  fpr <- fp/(fp+tn)
  #FNR
  fnr <- fn/(tp+fn)
  #LR+
  lr_pos <- tpr/fpr
  #LR-
  lr_neg <- fnr/tpr
  #FDR
  fdr <- fp/(fp+tp)
  
  #Summary
  cf_matrix <- matrix(c(tp, fn, fp, tn), dimnames=list(c("Predicted Positive", "Predicted Negative"),c("True Positive", "True Negative")),nrow = 2)
  print(cf_matrix)
  print(paste("Sensitivity:", round(tpr*100, 2), "%"))
  print(paste("Specificity:", round(tnr*100, 2), "%"))
  print(paste("Accuracy:", round(acc*100, 2), "%"))
  print(paste("Precision:", round(ppv*100, 2), "%"))
  print(paste("FDR:", round(fdr*100, 2), "%"))
  print(paste("DOR:", round(dor, 2)))
  print(paste("F1-score:", round(f1, 2)))
  print(paste("Expected Random TP ratio:", round(100*tp/((tp+fp)*0.00092), 0), "%"))
}

grn_eval(orig_graph, training_graph, output_graph, mode="train_asymetric", threshold = 1)
grn_eval(orig_graph, training_graph, output_graph, mode="test_asymetric", threshold = 1)

build_sym_adj <- function(adj_matrix) {
  adj_matrix_sym = adj_matrix + t(adj_matrix)
  return(adj_matrix_sym)
}
sym_output <- build_sym_adj(output_graph)
sym_orig <- build_sym_adj(orig_graph)
sym_training <- build_sym_adj(training_graph)

library(igraph)
plot(graph_from_adjacency_matrix(as.matrix(output_graph[1:10, 1:10]), mode="undirected", weighted=NULL))
plot(graph_from_adjacency_matrix(as.matrix(orig_graph[1:10, 1:10]), mode="undirected", weighted=NULL))

heatmap(as.matrix(orig_graph[1:100, 1:100]), Rowv=NA, Colv=NA, symm=T, scale="none")
heatmap(as.matrix(training_graph[1:100, 1:100]), Rowv=NA, Colv=NA, symm=T, scale="none")
heatmap(as.matrix(output_graph[1:100, 1:100]), Rowv=NA, Colv=NA, symm=T, scale="none")
