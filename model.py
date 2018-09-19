from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from keras.models import Model
from keras.layers import Dense, Input, Embedding, Dropout, Activation, Reshape, Flatten
from keras.layers.merge import concatenate, dot, add, multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.regularizers import l1, l2, l1_l2
from keras.initializers import RandomUniform
from keras.optimizers import RMSprop, Adam, SGD
from sklearn.metrics import roc_auc_score
import keras.backend as K
from nn_generator import DataGenerator
from keras.utils import multi_gpu_model
import tensorflow as tf
#from tensorflow.metrics import auc
def auc(y_true,y_pred):
    """ ROC AUC Score.
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.
    """
    with tf.name_scope("RocAucScore"):

        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)

        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p     = 3

        difference = tf.zeros_like(pos * neg) + pos - neg - gamma

        masked = tf.boolean_mask(difference, difference < 0.0)

    return tf.reduce_sum(tf.pow(-masked, p)) 
## define the model
def FunctionalDense(n, x, batchnorm=False, act='relu', lw1=0.0, dropout=0, name=''):
    if lw1 == 0.0:
        x = Dense(n, name=name+'_dense')(x)
    else:
        x = Dense(n, kernel_regularizer=l1(lw1), name=name+'_dense')(x)
    
    if batchnorm:
        x = BatchNormalization(name=name+'_batchnorm')(x)
        
    if act in {'relu', 'tanh', 'sigmoid'}:
        x = Activation(act, name=name+'_activation')(x)
    elif act =='prelu':
        x = PReLU(name=name+'_activation')(x)
    elif act == 'leakyrelu':
        x = LeakyReLU(name=name+'_activation')(x)
    elif act == 'elu':
        x = ELU(name=name+'_activation')(x)
    
    if dropout > 0:
        x = Dropout(dropout, name=name+'_dropout')(x)
        
    return x


def get_dot_model(K,adcateg_s,usrcateg_s,usrcateg_m,usrnumer_s,adnumer_s,namelist=[],
              lw=1e-4, lw1=1e-4, lr=1e-3, act='relu', batchnorm=False):
    val_bound=0.005
    inputindex=0
    embedding_inputs = []
    embedding_outputs = []

    
   
    if len(namelist) < len(adcateg_s+usrcateg_s+usrcateg_m):
        namelist  = [str(i) for i in list(range(len(adcateg_s+usrcateg_s+usrcateg_m)))]
        

    for i in range(len(adcateg_s+usrcateg_s)):
        feat = (adcateg_s+usrcateg_s)[i]
        name = namelist[inputindex]
     
            
        if len(feat.shape) > 1 and feat.shape[1]>1:
            
            tmp_input = Input(shape=(feat.shape[1],),name=name+"_oh_input")
            embedding_inputs.append(tmp_input)
            embedding_outputs.append(tmp_input)
        else:
            tmp_input = Input(shape=(1,),name=name+"_input")
            tmp_embeddings = Embedding(int(feat.max()+1),
                                       K if i!=0 else K*3,
                                       embeddings_initializer=RandomUniform(minval=-val_bound, maxval=val_bound),
                                       embeddings_regularizer=l2(lw if i!=0 else 0),
                                       input_length=1,
                                       trainable=True,
                                       name=name+"_embeddings")(tmp_input)
            tmp_embeddings = Flatten(name=name+'_flatten')(tmp_embeddings)
            embedding_inputs.append(tmp_input)
            embedding_outputs.append(tmp_embeddings)
        inputindex+=1
        
    
    for i in range(len(usrcateg_m)):
        feat = usrcateg_m[i]
        name = namelist[inputindex]
        if len(feat.shape) > 1 and feat.max() == 1:
            tmp_input = Input(shape=(feat.shape[1],),name=name+"_oh_input")
            embedding_inputs.append(tmp_input)
            embedding_outputs.append(tmp_input)
        else:
            tmp_input = Input(shape=(feat.shape[1],),name=name+"_input")
            tmp_embeddings = Embedding(int(feat.max()+1),
                                       K,
                                       embeddings_initializer=RandomUniform(minval=-val_bound, maxval=val_bound),
                                       embeddings_regularizer=l2(lw),
                                       input_length=feat.shape[1],
                                       trainable=True,
                                       name=name+"_embeddings")(tmp_input)
            tmp_embeddings = Flatten(name=name+'_flatten')(tmp_embeddings)
            embedding_inputs.append(tmp_input)
            embedding_outputs.append(tmp_embeddings)
        inputindex+=1
        
   
    #usr context 
    usr_context_input = Input(shape=(1 if type(usrnumer_s.shape) is int else usrnumer_s.shape[1],),name="usr_context_input")
    usr_profile = concatenate(embedding_outputs[len(adcateg_s):] + [usr_context_input],name='usr_profile')
    ad_context_input = Input(shape=(1 if type(adnumer_s.shape) is int else adnumer_s.shape[1],),name="ad_context_input")
    ad_profile = concatenate(embedding_outputs[1:len(adcateg_s)]+[ad_context_input],name='ad_profile')
    
#     #context 
#     context = []
#     for i in range(len(usr_embedding_features_single)):
#         for j in range(len(ad_embedding_features_single)):
#             tmpdot = dot([embedding_inputs[i], embedding_inputs[len(usr_embedding_features_single)+j]],
#                          axes=1, normalize=False, name='%s_%s_dot' % (usr_embedding_features_single[i],
#                                                                       ad_embedding_features_single[j]))
#             context.append(tmpdot)
        
#     dot_profile = concatenate(context,name='dot_profile')
    #dot_profile = add(context,name='dot_profile_add')
    
    #dot_profile    
    # user field
    #usr_embeddings = FunctionalDense(K*2, usr_profile, lw1=lw1, batchnorm=batchnorm, act=act, name='usr_profile_dense')
    #usr_embeddings = add([usr_embeddings,embedding_outputs[0]],name="usr_embeddings_addaid")
    usr_embeddings = Dense(K*3, name='usr_profile_linear')(usr_profile)
    #usr_embeddings = add([usr_embeddings, embedding_outputs[0]], name='usr_embeddings')
    #ad_embeddings = FunctionalDense(K*2, ad_profile, lw1=lw1, batchnorm=batchnorm,  act=act, name='ad_profile_dense')
    ad_embeddings = Dense(K*3, kernel_regularizer=l2(0.0001),name='ad_profile_linear')(ad_profile)
    usraddot = dot([usr_embeddings,ad_embeddings],axes=1,normalize=False,name='usr_ad_dot')
    
    #usraddot2 = dot([embedding_outputs[0],ad_embeddings],axes=1,normalize=False,name='usr_ad_dot2')
    #usradmultiply = Multiply([ad_embeddings,usr_embeddings])
    #dot_embeddings = FunctionalDense(K*2, dot_profile, lw1=lw1, batchnorm=batchnorm, act=act, name='dot_profile')
    #multiply([embedding_outputs[0],multiply])
    
    #joint = dot([usr_embeddings, ad_embeddings], axes=1, normalize=False, name='pred_cross')
    joint_embeddings = concatenate([ad_embeddings,embedding_outputs[0],usr_embeddings,usraddot], name='joint_embeddings')
    
    preds0 = FunctionalDense(K*3, joint_embeddings, batchnorm=batchnorm, act=act, name='preds_0')
    
    preds1 = FunctionalDense(K*3, concatenate([joint_embeddings, preds0]), batchnorm=batchnorm, act=act, name='preds_1')
    preds2 = FunctionalDense(K*2, concatenate([joint_embeddings, preds0, preds1]), batchnorm=batchnorm, act=act, name='preds_2')
#    preds3 = FunctionalDense(K, concatenate([joint_embeddings, preds0, preds1,preds2]), batchnorm=batchnorm, act=act, name='preds_3')
    
    preds = concatenate([joint_embeddings, preds0, preds1, preds2], name='prediction_aggr')
    
    preds = Dropout(0.5, name='prediction_dropout')(preds)
    
    preds = Dense(1,name='linear')(preds)
    #preds = add([dot_profile,preds],name='linear_deepwide')
    preds = Activation('sigmoid', name='prediction')(preds)
    #preds = Dense(1, activation='sigmoid', name='prediction')(preds)
    

    model = Model(inputs=embedding_inputs+[usr_context_input,ad_context_input], outputs=preds)
    #model = multi_gpu_model(model, gpus=2)

    opt = RMSprop(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    #model.compile(loss=auc, optimizer=opt)
    

    #model.compile(loss='mse', optimizer=opt)

    return model

def get_model(K,adcateg_s,usrcateg_s,usrcateg_m,usrnumer_s,adnumer_s,context,namelist=[],
              lw=1e-4, lw1=1e-4, lr=1e-3, act='relu', batchnorm=False):
    val_bound=0.005
    inputindex=0
    embedding_inputs = []
    embedding_outputs = []

    
   
    if len(namelist) < len(adcateg_s+usrcateg_s+usrcateg_m):
        namelist  = [str(i) for i in list(range(len(adcateg_s+usrcateg_s+usrcateg_m)))]
        

    for i in range(len(adcateg_s+usrcateg_s)):
        feat = (adcateg_s+usrcateg_s)[i]
        name = namelist[inputindex]
        
        if len(feat.shape) > 1 and feat.shape[1]>1:
            
            tmp_input = Input(shape=(feat.shape[1],),name=name+"_oh_input")
            embedding_inputs.append(tmp_input)
            embedding_outputs.append(tmp_input)
        else:
            tmp_input = Input(shape=(1,),name=name+"_input")
            tmp_embeddings = Embedding(int(feat.max()+1),
                                       K,
                                       embeddings_initializer=RandomUniform(minval=-val_bound, maxval=val_bound),
                                       embeddings_regularizer=l2(lw),
                                       input_length=1,
                                       trainable=True,
                                       name=name+"_embeddings")(tmp_input)
            tmp_embeddings = Flatten(name=name+'_flatten')(tmp_embeddings)
            embedding_inputs.append(tmp_input)
            embedding_outputs.append(tmp_embeddings)
        inputindex+=1
        

    for i in range(len(usrcateg_m)):
        feat = usrcateg_m[i]
        name = namelist[inputindex]
        if len(feat.shape) > 1 and feat.max() == 1:
            tmp_input = Input(shape=(feat.shape[1],),name=name+"_oh_input")
            embedding_inputs.append(tmp_input)
            embedding_outputs.append(tmp_input)
        else:
            tmp_input = Input(shape=(feat.shape[1],),name=name+"_input")
            tmp_embeddings = Embedding(int(feat.max()+1),
                                       K,
                                       embeddings_initializer=RandomUniform(minval=-val_bound, maxval=val_bound),
                                       embeddings_regularizer=l2(lw),
                                       input_length=feat.shape[1],
                                       trainable=True,
                                       name=name+"_embeddings")(tmp_input)
            tmp_embeddings = Flatten(name=name+'_flatten')(tmp_embeddings)
            embedding_inputs.append(tmp_input)
            embedding_outputs.append(tmp_embeddings)
        inputindex+=1
        
   
    #usr context 
    usr_context_input = Input(shape=(1 if type(usrnumer_s.shape) is int else usrnumer_s.shape[1],),name="usr_context_input")
    usr_profile = concatenate(embedding_outputs[len(adcateg_s):] + [usr_context_input],name='usr_profile')
    ad_context_input = Input(shape=(1 if type(adnumer_s.shape) is int else adnumer_s.shape[1],),name="ad_context_input")
    ad_profile = concatenate(embedding_outputs[:len(adcateg_s)]+[ad_context_input],name='ad_profile')
    #ad_id_profile = embedding_outputs[0]
    context_input = Input(shape=(1 if type(context.shape) is int else context.shape[1],),name="context_input")
    
#     #context 
#     context = []
#     for i in range(len(usr_embedding_features_single)):
#         for j in range(len(ad_embedding_features_single)):
#             tmpdot = dot([embedding_inputs[i], embedding_inputs[len(usr_embedding_features_single)+j]],
#                          axes=1, normalize=False, name='%s_%s_dot' % (usr_embedding_features_single[i],
#                                                                       ad_embedding_features_single[j]))
#             context.append(tmpdot)
        
#     dot_profile = concatenate(context,name='dot_profile')
    #dot_profile = add(context,name='dot_profile_add')
    
    #dot_profile    
    # user field
    #usr_embeddings = FunctionalDense(K*3, usr_profile, batchnorm=False, act=act, name='usr_profile_dense')
    #usr_embeddings = add([usr_embeddings,embedding_outputs[0]],name="usr_embeddings_addaid")
    #
    usr_embeddings = Dense(K*3, name='usr_profile_linear')(usr_profile)
    #usr_embeddings = add([usr_embeddings, embedding_outputs[0]], name='usr_embeddings')
    #ad_embeddings = FunctionalDense(K*2, ad_profile,  batchnorm=False,  act=act, name='ad_profile_dense')
    #
    ad_embeddings = Dense(K*3, name='ad_profile_linear')(ad_profile)
    
    #adid_embeddings = FunctionalDense(K*3, ad_id_profile, batchnorm=False, act=act, name='usr_profile_dense')
    #adid_embeddings = FunctionalDense(K*3, adid_embeddings, batchnorm=False, act=act, name='usr_profile_dense2')
    
    
   # context_embeddings = Dense(K*3,kernel_regularizer=l2(0.0001), name='context_dense')( context_input)
    usraddot = dot([usr_embeddings,ad_embeddings],axes=1,normalize=False,name='usr_ad_dot')
    #usraddot2 = dot([adid_embeddings,ad_embeddings],axes=1,normalize=False,name='usr_ad_dot2')
    
    
    #usraddot2 = dot([embedding_outputs[0],ad_embeddings],axes=1,normalize=False,name='usr_ad_dot2')
    
    
    #dot_embeddings = FunctionalDense(K*2, dot_profile, lw1=lw1, batchnorm=batchnorm, act=act, name='dot_profile')
    
   
    #joint = dot([usr_embeddings, ad_embeddings], axes=1, normalize=False, name='pred_cross')
    joint_embeddings = concatenate([usr_embeddings, ad_embeddings,usraddot], name='joint_embeddings')

    preds0 = FunctionalDense(K*3, joint_embeddings, batchnorm=batchnorm, act=act, name='preds_0')
    preds1 = FunctionalDense(K*3, concatenate([joint_embeddings, preds0]), batchnorm=batchnorm, act=act, name='preds_1')
    preds2 = FunctionalDense(K*2, concatenate([joint_embeddings, preds0, preds1]), batchnorm=batchnorm, act=act, name='preds_2')
    preds = concatenate([joint_embeddings, preds0, preds1, preds2], name='prediction_aggr')
    
    
    #preds0 = FunctionalDense(K*3, joint_embeddings, batchnorm=batchnorm, act=act, name='preds_0')
    #preds1 = FunctionalDense(K*3,  preds0, batchnorm=batchnorm, act=act, name='preds_1')
    #preds2 = FunctionalDense(K*2,  preds1, batchnorm=batchnorm, act=act, name='preds_2')
    preds = Dropout(0.5, name='prediction_dropout')(preds2)
    
    preds = Dense(1,name='linear')(preds)
    #preds = add([dot_profile,preds],name='linear_deepwide')
    preds = Activation('sigmoid', name='prediction')(preds)
    #preds = Dense(1, activation='sigmoid', name='prediction')(preds)
    

    model = Model(inputs=embedding_inputs+[usr_context_input,ad_context_input,context_input], outputs=preds)
    #model = Model(inputs=embedding_inputs+[usr_context_input,ad_context_input], outputs=preds)
    
    #model = multi_gpu_model(model, gpus=2)

    opt = RMSprop(lr=lr)
    model.compile(loss='binary_crossentropy',optimizer=opt)
    #model.compile(loss=auc, optimizer=opt)
    

    #model.compile(loss='mse', optimizer=opt)

    return model
def get_multiclass_model(K,usrcateg_s,usrcateg_m,usrnumer_s,namelist=[],
              lw=1e-4, lw1=1e-4, lr=1e-3, act='relu', batchnorm=False):
    val_bound=0.005
    inputindex=0
    embedding_inputs = []
    embedding_outputs = []

    
   
    if len(namelist) < len(usrcateg_s+usrcateg_m):
        namelist  = [str(i) for i in list(range(len(usrcateg_s+usrcateg_m)))]
        

    for i in range(len(usrcateg_s)):
        feat = (usrcateg_s)[i]
        name = namelist[inputindex]
        
        if len(feat.shape) > 1 and feat.shape[1]>1:
            
            tmp_input = Input(shape=(feat.shape[1],),name=name+"_oh_input")
            embedding_inputs.append(tmp_input)
            embedding_outputs.append(tmp_input)
        else:
            tmp_input = Input(shape=(1,),name=name+"_input")
            tmp_embeddings = Embedding(int(feat.max()+1),
                                       K,
                                       embeddings_initializer=RandomUniform(minval=-val_bound, maxval=val_bound),
                                       embeddings_regularizer=l2(lw),
                                       input_length=1,
                                       trainable=True,
                                       name=name+"_embeddings")(tmp_input)
            tmp_embeddings = Flatten(name=name+'_flatten')(tmp_embeddings)
            embedding_inputs.append(tmp_input)
            embedding_outputs.append(tmp_embeddings)
        inputindex+=1
        

    for i in range(len(usrcateg_m)):
        feat = usrcateg_m[i]
        name = namelist[inputindex]
        if len(feat.shape) > 1 and feat.max() == 1:
            tmp_input = Input(shape=(feat.shape[1],),name=name+"_oh_input")
            embedding_inputs.append(tmp_input)
            embedding_outputs.append(tmp_input)
        else:
            tmp_input = Input(shape=(feat.shape[1],),name=name+"_input")
            tmp_embeddings = Embedding(int(feat.max()+1),
                                       K,
                                       embeddings_initializer=RandomUniform(minval=-val_bound, maxval=val_bound),
                                       embeddings_regularizer=l2(lw),
                                       input_length=feat.shape[1],
                                       trainable=True,
                                       name=name+"_embeddings")(tmp_input)
            tmp_embeddings = Flatten(name=name+'_flatten')(tmp_embeddings)
            embedding_inputs.append(tmp_input)
            embedding_outputs.append(tmp_embeddings)
        inputindex+=1
    
    #usr context 
    usr_context_input = Input(shape=(1 if type(usrnumer_s.shape) is int else usrnumer_s.shape[1],),name="usr_context_input")
    usr_profile = concatenate(embedding_outputs + [usr_context_input],name='usr_profile')
    
    
    
    #usr_embeddings = FunctionalDense(K*2, usr_profile, lw1=lw1, batchnorm=batchnorm, act=act, name='usr_profile_dense')
    #usr_embeddings = add([usr_embeddings,embedding_outputs[0]],name="usr_embeddings_addaid")
    #usr_embeddings = Dense(K, name='usr_profile_linear')(usr_embeddings)
    #usr_embeddings = add([usr_embeddings, embedding_outputs[0]], name='usr_embeddings')
    #usraddot2 = dot([embedding_outputs[0],ad_embeddings],axes=1,normalize=False,name='usr_ad_dot2')

    #dot_embeddings = FunctionalDense(K*2, dot_profile, lw1=lw1, batchnorm=batchnorm, act=act, name='dot_profile')
    
   
    #joint = dot([usr_embeddings, ad_embeddings], axes=1, normalize=False, name='pred_cross')
    #joint_embeddings = concatenate([usr_profile,usr_embeddings], name='joint_embeddings')
    
    preds0 = FunctionalDense(K*5, usr_profile, batchnorm=batchnorm, act=act, name='preds_0')
    
    preds1 = FunctionalDense(K*3, preds0, batchnorm=batchnorm, act=act, name='preds_1')
    preds2 = FunctionalDense(K*2, preds1, batchnorm=batchnorm, act=act, name='preds_2')

    preds = Dropout(0.5, name='prediction_dropout')(preds2)
    preds = Dense(173,name='linear')(preds)
    #preds = add([dot_profile,preds],name='linear_deepwide')
    preds = Activation('softmax', name='prediction')(preds)
    model = Model(inputs=embedding_inputs+[usr_context_input], outputs=preds)
    opt = RMSprop(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    return model


