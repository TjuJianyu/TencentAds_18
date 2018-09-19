import model as md
import datetime
import importlib
import os
importlib.reload(md)
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
from nn_generator import DataGenerator
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import pandas as pd
from tqdm import tqdm
import numpy as np
import keras 
from collections import defaultdict
import gc
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
# train_nn list of np.array [np.array]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

single_features = ['uid','LBS','carrier','consumptionAbility','age','gender','house','education']
# multi_features = ["appIdAction","appIdInstall","ct","interest1","interest2","interest3",
#                   "interest4","interest5","kw1","kw2","kw3","os","topic1","topic2","topic3","marriageStatus"]
multi_features = ["ct","interest1","interest2","interest3",
                  "interest4","interest5","kw1","kw2","kw3","os","topic1","topic2","topic3","marriageStatus"]

ad_embedding_features_single = ['aid','advertiserId','campaignId','adCategoryId','productId','productType'] #'creativeId'
ad_numeric=['creativeSize']
usr_embedding_features_single= ['LBS','carrier','gender','education','house','age','consumptionAbility']
usr_numeric = ['house','age','consumptionAbility']

def loaddata():
    
    train = pd.read_csv("data/preliminary_contest_data/train.csv",engine='c')
    test = pd.read_csv("data/preliminary_contest_data/test1.csv",engine='c')
    adattrs = pd.read_csv("data/preliminary_contest_data/adFeature.csv",engine='c')
    usersingle = pd.read_csv("data/preliminary_contest_data/unstack_single_train.csv",engine='c')
    train_nn = []
    for feat in tqdm(multi_features):
        tmp_trnn = pd.read_csv("data/preliminary_contest_data/%s_nn_train.csv" % feat,engine='c')
        tmp_trnn.fillna(0,inplace=True)
        
        tmp_trnn = tmp_trnn.astype('int32')
        train_nn.append(tmp_trnn.values)
        del tmp_trnn
    # encoding 
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for feat in ['advertiserId','campaignId','creativeId','adCategoryId','productId','productType']:
        adattrs[feat] = le.fit_transform(adattrs[feat])
    #merge all data together
    trainmerge = train.merge(adattrs,on='aid',how='left')
    trainmerge = trainmerge.merge(usersingle,on='uid',how='left')
    trainmerge['label'] = (trainmerge['label']+1)/2
    for feat in ['aid']:
        trainmerge[feat] = le.fit_transform(trainmerge[feat])  
    trainmerge.fillna(0,inplace=True)
    trainmerge = trainmerge.astype('int32')
#     usercount = []
#     for i in tqdm(range(len(multi_features))):
#          usercount.append((train_nn[i].values!=0).sum(axis=1))
#     usercount = np.array(usercount).T
    del adattrs,usersingle
    return trainmerge,train_nn
def load_testdata():
    
    train = pd.read_csv("data/preliminary_contest_data/train.csv",engine='c')
    test = pd.read_csv("data/preliminary_contest_data/test1.csv",engine='c')
    adattrs = pd.read_csv("data/preliminary_contest_data/adFeature.csv",engine='c')
    usersingle = pd.read_csv("data/preliminary_contest_data/unstack_single_train.csv",engine='c')
    train_nn = []
    for feat in tqdm(multi_features):
        tmp_trnn = pd.read_csv("data/preliminary_contest_data/%s_nn_test.csv" % feat)
        tmp_trnn.fillna(0,inplace=True)
        tmp_trnn = tmp_trnn.values
        tmp_trnnmax = tmp_trnn.max() 
        
        
        if tmp_trnnmax < 255:
            tmp_trnn.astype('uint8')
        elif tmp_trnnmax < 65535:
            tmp_trnn.astype('uint16')
        elif tmp_trnnmax < 2 ** 32 - 1:
            tmp_trnn.astype('uint32')
        else:
            tmp_trnn.astype('uint64')
        train_nn.append(tmp_trnn)
        del tmp_trnn
        gc.collect()
    # encoding 
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for feat in ['advertiserId','campaignId','creativeId','adCategoryId','productId','productType']:
        adattrs[feat] = le.fit_transform(adattrs[feat])
    #merge all data together
    trainmerge = test.merge(adattrs,on='aid',how='left')
    trainmerge = trainmerge.merge(usersingle,on='uid',how='left')
    #trainmerge['label'] = (trainmerge['label']+1)/2
    for feat in ['aid']:
        trainmerge[feat] = le.fit_transform(trainmerge[feat])  
    trainmerge.fillna(0,inplace=True)
    trainmerge = trainmerge.astype('int32')
#     usercount = []
#     for i in tqdm(range(len(multi_features))):
#          usercount.append((train_nn[i].values!=0).sum(axis=1))
#     usercount = np.array(usercount).T
    del adattrs,usersingle
    return trainmerge,train_nn

def multifeat_onehot(train_nn,count_threshold = 20,):
    ohe = OneHotEncoder(sparse=False)
    train_nn_onehot = []
    onehot_index = []
    for i in tqdm(range(len(multi_features))):
        train_nnmax = train_nn[i].max()
        if  train_nnmax < count_threshold:
            tmpohe = ohe.fit_transform(train_nn[i].reshape((len(train_nn[i])*train_nn[i].shape[1],1)))[:,1:].reshape((len(train_nn[i]),train_nn[i].shape[1],train_nnmax)).sum(axis=1)
            train_nn_onehot.append(tmpohe.astype(int))
            onehot_index.append(i)
    train_nn_index = list(set(list(range(len(multi_features))))- set(onehot_index))
    return train_nn_onehot, onehot_index

def train_valid_split_multi(train_nn,train_nn_onehot=None,test_size=0.2,random_state=42):
    offttr_nn = []
    offtte_nn = []
    for feat in tqdm(range(len(multi_features))):
        a,b = train_test_split(train_nn[feat],test_size=test_size,random_state=random_state)
        offttr_nn.append(a)
        offtte_nn.append(b)

    offttr_nn_onehot = []
    offtte_nn_onehot = []
    if train_nn_onehot is not None:
        for feat in tqdm(range(len(train_nn_onehot))):
            a,b = train_test_split(train_nn_onehot[feat],test_size=test_size,random_state=random_state)
            offttr_nn_onehot.append(a)
            offtte_nn_onehot.append(b)
    return offttr_nn,offtte_nn,offttr_nn_onehot,offtte_nn_onehot
        
        
def train_valid_split_single(trainmerge,single_count_threshold = 200,test_size=0.2,random_state=42):
    suboffttr,subofftte = train_test_split(trainmerge,test_size=test_size,random_state=random_state)
    ohe = OneHotEncoder(sparse=False)
    train_embeddings = []
    valid_embeddings = []
    for feat in ad_embedding_features_single+usr_embedding_features_single:
        if int(trainmerge[feat].max()+1)< single_count_threshold:
            train_embeddings.append(ohe.fit_transform(suboffttr[[feat]].values))
            valid_embeddings.append(ohe.transform(subofftte[[feat]].values))
        else:    
            train_embeddings.append(suboffttr[feat].values)
            valid_embeddings.append(subofftte[feat].values)
    ss_context = StandardScaler()
    usr_train_context = ss_context.fit_transform(suboffttr[usr_numeric].values)
    usr_valid_context = ss_context.transform(subofftte[usr_numeric].values)
    ad_train_context = ss_context.fit_transform(suboffttr[ad_numeric].values)
    ad_valid_context = ss_context.transform(subofftte[ad_numeric].values)
    #     usr_train_count = ss_context.fit_transform(suboffttr_count)
    #     usr_valid_count = ss_context.fit_transform(subofftte_count)
    return train_embeddings,valid_embeddings, usr_train_context,usr_valid_context,ad_train_context,ad_valid_context,suboffttr['label'],subofftte['label']
#offttr_count,offtte_count = train_test_split(usercount,test_size=0.2,random_state=443)
def takesample(trainmerge,train_nn ,rate=0.2,random_state=42):
    _,subtrainmerge = train_test_split(trainmerge,test_size=rate,random_state = random_state)
    subtrainmerge = subtrainmerge.reset_index(drop=True)
    subtrain_nn = []
    for feat in tqdm(train_nn):
        _,sub_nn = train_test_split(feat.values,test_size=rate,random_state = random_state)
        subtrain_nn.append(sub_nn)
    
    return subtrainmerge, subtrain_nn

def taketest(trainmerge,train_nn ,rate=0.2,random_state=42):
    subtrainmerge,testmerge = train_test_split(trainmerge,test_size=rate,random_state = random_state)
    subtrainmerge = subtrainmerge.reset_index(drop=True)
    testmerge = testmerge.reset_index(drop=True)
    subtrain_nn = []
    subtest_nn = []
    for feat in tqdm(train_nn):
        test_nn,sub_nn = train_test_split(feat,test_size=rate,random_state = random_state)
        subtrain_nn.append(sub_nn)
        subtest_nn.append(test_nn)
    return subtrainmerge, subtrain_nn,testmerge, subtest_nn
def downsample(trainmerge,train_nn):
    pos_trainmerge = trainmerge[trainmerge['label']==1]
    pos_count = len(pos_trainmerge)
    print(pos_count)
    neg_trainmerge = trainmerge[trainmerge['label'] == 0].sample(pos_count)
    subtrainmerge = pd.concat((pos_trainmerge,neg_trainmerge)).reset_index(drop=True)
    subtrain_nn = []
    for feat in tqdm(train_nn):
        subtrain_nn.append(pd.concat((feat.iloc[pos_trainmerge.index],feat.iloc[neg_trainmerge.index])).reset_index(drop=True).values)
    return subtrainmerge, subtrain_nn

def batch_train_test_split(tosplit,kfold=None,test_size=0.2,random_state=42):
    trainlist = []
    testlist = []
    for val in tqdm(tosplit):
        train,test = train_test_split(val,test_size=0.2,random_state=random_state,shuffle=True)
        trainlist.append(train)
        testlist.append(test)
    return trainlist,testlist

def batch_s_onehot(toonehot,toonehot_transform=None,count_threshold = None,toonehotindex=None):
    assert (count_threshold is not None or toonehotindex is not None) and not \
    (count_threshold is not None and toonehotindex is not None),"count_threshold and\
    toonehotindex can not all None, can not all not None"
    assert toonehot_transform is None or (toonehot_transform is not None and  len(toonehot_transform) == len(toonehot)),"len(toonehot_transform) should equal to len(toonehot)"
    ohe = OneHotEncoder(sparse=False)
    
    if count_threshold  is not None:
        for index in tqdm(range(len(toonehot))):
            if toonehot[index].max()<count_threshold:
                toonehot[index] = ohe.fit_transform(toonehot[index][:,np.newaxis])
                if toonehot_transform is not None:
                    toonehot_transform[index] = ohe.transform(toonehot_transform[index][:,np.newaxis] )
    elif toonehotindex is not None:
        for index in toonehotindex:
            toonehot[index] = ohe.fit_transform(toonehot[index][:,np.newaxis])
            if toonehot_transform is not None:
                toonehot_transform[index] = ohe.transform(toonehot_transform[index][:,np.newaxis])
    
    return toonehot,toonehot_transform
    
    


class auc_callback(Callback):
    def __init__(self,x,xn,y):
        self.x = x
        self.xn = xn
        self.y = y
    def on_train_begin(self, logs={}):
        if not ('val_auc_loss' in self.params['metrics']):
            self.params['metrics'].append('val_auc_loss')

    def on_epoch_end(self, epoch, logs={}):
        
        dataGenerator = DataGenerator()
        flow = dataGenerator.flow(self.x,self.xn, batch_size=16384, shuffle=False)
        y_pred = self.model.predict_generator(flow, flow.__len__(), workers=4)
        
        roc = roc_auc_score(self.y, y_pred)
        print("roc_auc %.6f\n" % roc)
        logs['val_auc_loss'] = roc
        

        
trainmerge,train_nn = loaddata()
gc.collect()

seed = np.random.randint(0,100000)
sser= StandardScaler()
usrcateg_s_train,usrcateg_s_valid = batch_train_test_split([trainmerge[feat].values for feat in 
                                                      usr_embedding_features_single],
                                                     test_size=0.2,random_state=seed)

adcateg_s_train,adcateg_s_valid = batch_train_test_split([trainmerge[feat].values for feat in 
                                                      ad_embedding_features_single],
                                                     test_size=0.2,random_state=seed)
adcateg_s_train,adcateg_s_valid = batch_s_onehot(adcateg_s_train,adcateg_s_valid,count_threshold=200)


usrnumer_s_train,usrnumer_s_valid = batch_train_test_split([trainmerge[feat].values for feat in usr_numeric ],
                                                     test_size=0.2,random_state=seed)
usrnumer_s_train = np.array(usrnumer_s_train).T
usrnumer_s_valid = np.array(usrnumer_s_valid).T
usrnumer_s_train = sser.fit_transform(usrnumer_s_train)
usrnumer_s_valid = sser.transform(usrnumer_s_valid)
adnumer_s_train,adnumer_s_valid = batch_train_test_split([trainmerge[feat].values for feat in ad_numeric ],
                                                     test_size=0.2,random_state=seed)
adnumer_s_train = np.array(adnumer_s_train).T
adnumer_s_valid = np.array(adnumer_s_valid).T
adnumer_s_train = sser.fit_transform(adnumer_s_train)
adnumer_s_valid = sser.transform(adnumer_s_valid)
usrcateg_m_train,usrcateg_m_valid = batch_train_test_split(train_nn,
                                                     test_size=0.2,random_state=seed)
m_oht_nn, m_oht_nnindex = multifeat_onehot(train_nn,count_threshold = 20,)
usrcateg_m_oh_train,usrcateg_m_oh_valid= batch_train_test_split(m_oht_nn,
                                                     test_size=0.2,random_state=seed)

y_train,y_valid = batch_train_test_split([trainmerge['label'].values],test_size=0.2,random_state=seed)
y_train = y_train[0]
y_valid = y_valid[0]
aid_train,aid_valid = batch_train_test_split([trainmerge['aid'].values],test_size=0.2,random_state=seed)
aid_train = aid_train[0]
aid_valid = aid_valid[0]
del train_nn
gc.collect()
        
#para = pd.read_csv('./nn_record.csv').sort_values(by='val_auc', ascending=False)
for i in range(100):
    K = np.random.randint(4, 128)  # 64
    lw = 7.5e-4 * (0.1 ** (np.random.rand() * 3 - 1.5))
    lw1 =  1e-3 * (0.1 ** (np.random.rand() * 3 - 1.5))
    lr = 7.5e-4 * (0.1 ** (np.random.rand() * 2 - 1.0))
    lr_decay = 0.65 + np.random.rand() * 0.3
    #activation = np.random.choice(['relu', 'tanh', 'prelu', 'leakyrelu', 'elu'])
    #batchnorm = np.random.choice([True, False])
    activation = 'relu'
    batchnorm = True

    sample_weight_rate = 0.0
#     lr = 0.0014
#     lr_decay = 0.57
#     lw1=0.001
#     lw = 0.001
#     K = 60
    epoch=40
    batch_size=8196

    #modelname =  "%f_%f_%f_%f_%d_%d_%d_%s_%d" % (lr,lr_decay,lw1,lw,K,epoch,batch_size,"withouttwoapp",seed,)
    print('K: %d, lw: %e, lw1: %e, lr: %e, lr_decay: %f, act: %s, batchnorm: %s'%(K, lw,lw1, lr, lr_decay, activation, batchnorm))
    model = md.get_model(K,adcateg_s_train,usrcateg_s_train,usrcateg_m_train,usrnumer_s_train,adnumer_s_train,
                         lw=lw,lr=lr,lw1=lw1,act=activation,batchnorm=batchnorm)

    dataGenerator = DataGenerator()
    train_flow = dataGenerator.flow(adcateg_s_train+usrcateg_s_train+usrcateg_m_train,
                                    [usrnumer_s_train,adnumer_s_train],
                                    y=y_train,
                                    batch_size=batch_size, shuffle=True)
    valid_flow = dataGenerator.flow(adcateg_s_valid+usrcateg_s_valid+usrcateg_m_valid,
                                    [usrnumer_s_valid,adnumer_s_valid],
                                    y=y_valid,
                                    batch_size=batch_size, shuffle=False)
    early_stopping =EarlyStopping(monitor='val_auc_loss', patience=2, min_delta=0.0000,verbose=2,mode='max')
    
    timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    modelseed =  np.random.randint(0,65536)
    model_path = 'model/model_1/bst_model_%s_%d.h5'%(timenow,modelseed)
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_auc_loss', save_best_only=True, save_weights_only=True)
    lr_reducer = LearningRateScheduler(lambda x: lr*(lr_decay**x))
    hist = model.fit_generator(train_flow, train_flow.__len__(), 
                               epochs=epoch, workers=4,#class_weight={1:20,0:1},
                               callbacks=[lr_reducer,
                                          #RocAucMetricCallback(predict_batch_size=batch_size),
                                          auc_callback(adcateg_s_valid+usrcateg_s_valid+usrcateg_m_valid,
                                                      [usrnumer_s_valid,adnumer_s_valid],
                                                      y_valid),early_stopping, model_checkpoint,
                                         ],
                               validation_data = valid_flow,
                               validation_steps=valid_flow.__len__())

    model.load_weights(model_path)
    
    bst_epoch = np.argmax(hist.history['val_auc_loss'])
    trn_loss = hist.history['loss'][bst_epoch]
    val_loss = hist.history['val_loss'][bst_epoch]
    val_auc_loss = hist.history['val_auc_loss'][bst_epoch]

#     valid_flow = dataGenerator.flow(adcateg_s_valid+usrcateg_s_valid+usrcateg_m_valid,
#                                     [usrnumer_s_valid,adnumer_s_valid],
#                                     y=y_valid,
#                                     batch_size=batch_size, shuffle=False)
#    model.save_weights("model/weights_%s.h5" % modelname)
#     valid_pred = model.predict_generator(valid_flow, valid_flow.__len__(), workers=1)
#     roc_auc_score(y_valid,valid_pred[:,0])

    res = '%s,%s,%d,%d,%s,%s,%d,%e,%e,%e,%f,%e,%.5f,%.5f,%.5f,\n'%(timenow, \
            'model_1',modelseed,seed, activation, batchnorm, K, lw, lw1, lr, lr_decay,  bst_epoch+1, trn_loss, \
            val_loss, val_auc_loss)
    f = open('./nn_record.csv', 'a')
    f.write(res)
    f.close()
    
