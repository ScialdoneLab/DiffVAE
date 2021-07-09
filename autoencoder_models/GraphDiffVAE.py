from keras.layers import Input, Lambda, Average, Concatenate
from keras.models import Model, Sequential
from keras import metrics, optimizers
from keras import backend as K
from keras.regularizers import l2
from keras.metrics import AUC
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import LambdaCallback, ModelCheckpoint
from sklearn.metrics import roc_auc_score

from autoencoder_models.base.base_VAE import BaseVAE
from autoencoder_models.base.gcn_layers import GraphConvolution, InnerProduct

import time


class GraphDiffVAE(BaseVAE):
    def __init__(self, num_nodes, num_features, adj_matrix, latent_dim, hidden_layers_dim, epochs, learning_rate, loss_mode, model_select, kl_weight, timestamp, adj_val):
        BaseVAE.__init__(self, original_dim=None, latent_dim=latent_dim,
                         batch_size=1, epochs=epochs, learning_rate=learning_rate)
        self.hidden_layers_dim = hidden_layers_dim
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.adj_matrix = adj_matrix
        self.model_select = model_select
        self.kl_weight = kl_weight
        self.model_fp = ('results/Models/' + timestamp + '_')
        self.beta = K.variable(value=0.0)
        self.adj_val = K.variable(adj_val)
        if loss_mode == 'binary':
            self.recon_loss = self.recon_loss_binary
        elif loss_mode == 'categorical':
            self.recon_loss = self.recon_loss_categorical
        elif loss_mode == 'f1':
            self.recon_loss = self.f1_loss_smooth

        self.build_encoder()
        self.build_decoder()
        self.compile_vae()

    def build_encoder(self):
        # Input placeholder
        self.input_placeholder = Input(shape=(self.num_features,))
        self.encoder_layer = self.input_placeholder

        encoder_hidden_layer = GraphConvolution(output_dim=self.hidden_layers_dim[0],
                                         adj_matrix=self.adj_matrix,
                                         activation='LeakyReLU')(self.encoder_layer)


        self.z_mean = GraphConvolution(output_dim=self.latent_dim,
                                       adj_matrix=self.adj_matrix,
                                       activation='linear')(encoder_hidden_layer)


        self.z_log_var = GraphConvolution(output_dim=self.latent_dim,
                                          adj_matrix=self.adj_matrix,
                                          activation='linear', kernel_initializer='zeros')(encoder_hidden_layer)


        self.z = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])

        self.encoder = Model(self.input_placeholder, self.z_mean)

    def build_decoder(self):

        features_concat = Concatenate()([self.encoder_layer, self.z])

        decoder_hidden_layer = GraphConvolution(output_dim=self.hidden_layers_dim[0],
                                                adj_matrix=self.adj_matrix,
                                                activation='LeakyReLU')(features_concat)

        decoder_z = GraphConvolution(output_dim=self.latent_dim,
                                                adj_matrix=self.adj_matrix,
                                                activation='linear')(decoder_hidden_layer)

        z_combined = Average()([self.z, decoder_z]) # lambda = 0.5
        self.x_decoded_mean = InnerProduct(num_nodes=self.num_nodes)(z_combined)

    @tf.function
    def graph_vae_loss_function(self, x, x_decoded_mean):

        # tf.print("z_log_var", K.int_shape(self.z_log_var))

        recon_loss = self.recon_loss(x, x_decoded_mean)
        kl_loss = self.kl_loss(x, x_decoded_mean)

        total_loss = recon_loss + self.kl_weight * self.beta * kl_loss

        return total_loss

    def recon_loss_binary(self, x, x_decoded_mean):
        x = K.reshape(x, shape=(self.num_nodes*self.num_nodes,1))
        x_decoded_mean = K.reshape(x_decoded_mean, shape=(self.num_nodes*self.num_nodes,1))
        norm = self.adj_matrix.shape[0] * self.adj_matrix.shape[0] / \
               float((self.adj_matrix.shape[0] * self.adj_matrix.shape[0] -
                      self.adj_matrix.sum()) * 2)
        recon_loss_binary = norm * K.sum(metrics.binary_crossentropy(y_true=x, y_pred=x_decoded_mean))
        return recon_loss_binary

    def recon_loss_categorical(self, x, x_decoded_mean):
        x = K.reshape(x, shape=(self.num_nodes*self.num_nodes,))
        x_decoded_mean = K.reshape(x_decoded_mean, [-1])
        norm = self.adj_matrix.shape[0] * self.adj_matrix.shape[0] / \
               float((self.adj_matrix.shape[0] * self.adj_matrix.shape[0] -
                      self.adj_matrix.sum()) * 2)
        recon_loss_cat = recon_loss = norm * metrics.categorical_crossentropy(y_true=x, y_pred=x_decoded_mean)
        return recon_loss_cat

    def kl_loss(self, x, x_decoded_mean):
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return K.mean(kl_loss)

    def f1_loss_smooth(self, y_true, y_pred):

        tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
        tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
        fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2*p*r / (p+r+K.epsilon())
        f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
        return 1 - K.mean(f1)

    def kl_warmup(self, epoch):
        def h(x):
            return np.exp((-1)/x)
        if epoch == 0:
            value = 0
        elif epoch >= self.epochs/2:
            value = 1
        else:
            it = epoch*2/self.epochs
            value = h(it)/(h(it)+h(1-it))
        K.set_value(self.beta, value)
        return value

    def kl_weight_log(self, y_true, y_pred):
        return (self.beta * self.kl_weight)

    def val_auc(self, y_true, y_pred):
        y_true_vec = K.reshape(y_true, shape=(self.num_nodes*self.num_nodes,)) 
        y_pred_vec = K.reshape(y_pred, shape=(self.num_nodes*self.num_nodes,))
        y_adj_vec = K.reshape(self.adj_val, shape=(self.num_nodes*self.num_nodes,))
        mask = ~tf.cast(y_true_vec, tf.bool)
        y_true_vec_masked = tf.boolean_mask(y_adj_vec, mask)
        y_pred_vec_masked = tf.boolean_mask(y_pred_vec, mask)
        
        def roc_auc(adj_val, y_true, y_pred):
            try:
                return roc_auc_score(y_true=y_true, y_score=y_pred)
            except:
                return 0
        
        return tf.py_function(roc_auc, (self.adj_val, y_true_vec_masked, y_pred_vec_masked), tf.double)


    def train_auc(self, y_true, y_pred):
        y_true_vec = K.reshape(y_true, shape=(self.num_nodes*self.num_nodes,))
        y_pred_vec = K.reshape(y_pred, shape=(self.num_nodes*self.num_nodes,))
        
        def roc_auc(adj_val, y_true, y_pred):
            try:
                return roc_auc_score(y_true=y_true, y_score=y_pred)
            except:
                return 0
        
        return tf.py_function(roc_auc, (self.adj_val, y_true_vec, y_pred_vec), tf.double)


    def compile_vae(self):
        K.set_learning_phase(1)
        self.graph_vae = Model(self.input_placeholder, self.x_decoded_mean)

        adam_optimizer = optimizers.Adam(lr=self.learning_rate, clipvalue=0.5)
        metrics = [self.kl_loss, self.recon_loss]
        self.graph_vae.compile(optimizer=adam_optimizer, loss=self.graph_vae_loss_function, metrics=[self.train_auc, self.val_auc, self.kl_weight_log, self.kl_loss, self.recon_loss])
        self.mcp = ModelCheckpoint(filepath=(self.model_fp + 'weights.hdf5'), monitor='val_auc', verbose=0, save_best_only=True, save_weights_only=True, mode='min')
        print (self.graph_vae.summary())

    def train_vae(self, node_features):
        wu_cb = LambdaCallback(on_epoch_end=lambda epoch, log: self.kl_warmup(epoch))
        hist = self.graph_vae.fit(node_features, self.adj_matrix,
                epochs=self.epochs,
                batch_size=self.num_nodes,
                shuffle=False, verbose=2, callbacks = [wu_cb, self.mcp])
        if self.model_select == "best":
            self.graph_vae.load_weights(self.model_fp + 'weights.hdf5')

        # Plot training & validation metrics
        # print(hist.history.keys())
        figure, axis = plt.subplots(2,2)

        axis[0, 0].plot(hist.history['loss'])
        axis[0, 0].set_title('Total loss')
        axis[0, 0].set_xlabel('Epoch')

        axis[0, 1].plot(hist.history['kl_loss'])
        axis[0, 1].set_title('KL loss')
        axis[0, 1].set_xlabel('Epoch')
        axis1_1 = axis[0, 1].twinx()
        axis1_1.set_ylabel = "KL weight"
        axis1_1.plot(hist.history['kl_weight_log'], color='tab:orange')

        axis[1, 0].plot(hist.history[self.recon_loss.__name__])
        axis[1, 0].set_title('Recon loss')
        axis[1, 0].set_xlabel('Epoch')

        axis[1, 1].plot(hist.history['train_auc'])
        axis[1, 1].plot(hist.history['val_auc'])
        axis[1, 1].set_title('ROC AUC')
        axis[1, 1].set_xlabel('Epoch')

        figure.tight_layout()

        plt.savefig(self.model_fp + 'training_history.png', dpi=300)

        return self.graph_vae.predict(node_features, batch_size=self.num_nodes), \
               self.encoder.predict(node_features, batch_size=self.num_nodes)
