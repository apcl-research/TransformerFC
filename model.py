import tensorflow.keras as keras
import tensorflow as tf

from models.ast_attendgru_fc import AstAttentionGRUFCModel as ast_attendgru_fc
from models.codegnngru import CodeGNNGRUModel as codegnngru
from models.codegnngru_sep import CodeGNNGRUModelSep as codegnngru_sep
from models.code2seq import Code2SeqModel as code2seq
from models.transformer_base import TransformerBase as xformer_base
from models.setransformer import SeTransformer as sexformer
from models.transformer_FCimport TransformerBaseFC3 as xformer_base_fc3
from models.transformer_alt import TransformerBaseFC7 as xformer_base_fc7
from models.transformer_base_fccomb import TransformerBaseFCCombined as xformer_base_fccomb


# from models.attendgru_bio2 import AttentionGRUBio2Model as attendgru_bio2

def create_model(modeltype, config):
    mdl = None

    elif modeltype == 'codegnngru':
        mdl = codegnngru(config)
    elif modeltype == 'codegnngru-sep':
        mdl = codegnngru_sep(config)
    elif modeltype == 'ast-attendgru-fc':
        mdl = ast_attendgru_fc(config)
    elif modeltype == 'code2seq':
        mdl = code2seq(config)
    elif modeltype == 'transformer-base':
        mdl = xformer_base(config)
    elif modeltype == 'setransformer':
        mdl = sexformer(config)
    elif modeltype == 'transformer-base-fccomb':
        mdl = xformer_base_fccomb(config)
    elif modeltype == 'transformer-base-fc3':
        mdl = xformer_base_fc3(config)
    elif modeltype == 'transformer-base-fc7':
        mdl = xformer_base_fc7(config)
    else:
        print("{} is not a valid model type".format(modeltype))
        exit(1)
        
    return mdl.create_model()
