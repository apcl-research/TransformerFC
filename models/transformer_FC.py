import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, GRU, Concatenate, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, GlobalAveragePooling1D
from tensorflow.compat.v1.keras.layers import Bidirectional, RepeatVector, Permute, TimeDistributed, dot
from tensorflow.keras.optimizers import RMSprop, Adamax
from tensorflow.keras import utils, metrics

from custom.qstransformer_layers import TransformerBlock, TokenAndPositionEmbedding, MultiHeadAttentionBlock
from custom.qs_loss import use_prep, attendgru_prep, custom_use_loss,  custom_attendgru_loss, custom_cce_loss, custom_dist_cce_loss

class TransformerBaseFC3:
    def __init__(self, config):
        
        config['tdatlen'] = 50

        config['sdatlen'] = 20
        config['stdatlen'] = 25
        
        config['stackdepth'] = 3 # BERT and GPT-2 use 12, so why not?

        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.datlen = config['tdatlen']
        self.sdatlen = config['sdatlen']
        self.comlen = config['comlen']
        
        self.embdims = 128
        self.attheads = 3 # number of attention heads
        self.attheads_sdats = 3
        self.recdims = 128 
        self.ffdims = 128 # hidden layer size in feed forward network inside transformer

        self.config['batch_config'] = [ ['tdat', 'sdat', 'com'], ['comout'] ]
        self.config['loss_type'] = config['loss_type']
        if self.config['loss_type'] == 'use':
            self.index_tensor, self.comwords_tensor = use_prep(self.config['comstok'])
        elif self.config['loss_type'] == 'attendgru':
            self.fmodel = attendgru_prep()
        elif self.config['loss_type'] == 'use-dist':
            self.dist = config['target_dist']

    def create_model(self):
        
        dat_input = Input(shape=(self.datlen,))
        sdat_input = Input(shape=(self.sdatlen, self.config['stdatlen']))
        com_input = Input(shape=(self.comlen,))
        
        ee = TokenAndPositionEmbedding(self.datlen, self.tdatvocabsize, self.embdims)
        se = TokenAndPositionEmbedding(self.datlen, self.tdatvocabsize, self.embdims)

        see = TimeDistributed(se)
        seeout = see(sdat_input)
        seeout = Reshape((self.sdatlen*self.config['stdatlen'], self.embdims), input_shape=(self.sdatlen, self.config['stdatlen'], self.embdims))(seeout)
        
        setransformer_block = TransformerBlock(self.embdims, self.attheads_sdats, self.ffdims)
        #setransformer_block = TimeDistributed(setransformer_block)
        sencout = setransformer_block(seeout, seeout)

        eeout = ee(dat_input)
        etransformer_block = TransformerBlock(self.embdims, self.attheads, self.ffdims)
        encout = etransformer_block(eeout, eeout)
        #sencout = etransformer_block(seeout, seeout)
        
        for k in range(self.config['stackdepth'] - 1):
            etransformer_block = TransformerBlock(self.embdims, self.attheads, self.ffdims)
            #setransformer_block = TransformerBlock(self.embdims, self.attheads_sdats, self.ffdims)
            encout = etransformer_block(encout, encout)
            sencout = setransformer_block(sencout, sencout)
        
        #encout = concatenate([encout, sencout], axis=1)

        de = TokenAndPositionEmbedding(self.comlen, self.comvocabsize, self.embdims)
        deout = de(com_input)
        de_mha1 = MultiHeadAttentionBlock(self.embdims, self.attheads)
        de_mha1_out = de_mha1(deout, deout)
        dtransformer_block = TransformerBlock(self.embdims, self.attheads, self.ffdims)
        decout = dtransformer_block(de_mha1_out, sencout)

        #de2 = TokenAndPositionEmbedding(self.comlen, self.comvocabsize, self.embdims)
        #deout2 = de2(com_input)
        #de_mha12 = MultiHeadAttentionBlock(self.embdims, self.attheads)
        #de_mha1_out2 = de_mha1(deout, deout)
        dtransformer_block2 = TransformerBlock(self.embdims, self.attheads, self.ffdims)
        #decout2 = dtransformer_block2(de_mha1_out, sencout)
        decout2 = dtransformer_block2(decout, encout)

        #decout = concatenate([decout, decout2], axis=1)

        context = decout2
        out = context
        # out = TimeDistributed(Dense(self.recdims, activation="tanh"))(context)
        out = Flatten()(out)
        out = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[dat_input, sdat_input, com_input], outputs=out)
        lossf = custom_cce_loss()
        optimizer = keras.optimizers.Adam(lr=3e-4)
        if self.config['loss_type'] == 'use':
            lossf = custom_use_loss(self.index_tensor, self.comwords_tensor)
        elif self.config['loss_type'] == 'attendgru':
            lossf = custom_attendgru_loss(self.fmodel)
        elif self.config['loss_type'] == 'use-dist':
            lossf = custom_dist_cce_loss(self.dist)

        model.compile(loss=lossf, optimizer=optimizer, metrics=['accuracy'])
        return self.config, model
