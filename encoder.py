import pandas as pd 

from sklearn.preprocessing import OneHotEncoder
#from target_encoder import TargetEncoder

import sys
from sklearn.preprocessing import LabelEncoder
import numpy as np 
class ExLabelEncoder(object):
	"""docstring for LabelEncoder"""
	def __init__(self, fillna=None):
		#fillna: None(default) , "other" ,anythingyouwant
		#how to deal with NAN in label encoding processing default: None -> np.nan	
		self.fillna = fillna
		self.sklabelencoder= LabelEncoder()
		self.nan=chr(255)*5
		if self.fillna is not None and self.fillna is not 'other':
			self.nan = fillna
	# train pandas series (#example,)
	def fit(self,train):
		self.sklabelencoder.fit(train.fillna(self.nan))
		self.classes_ = self.sklabelencoder.classes_
	def transform(self,test):
		res = self.sklabelencoder.transform(test.fillna(self.nan))
		if self.fillna is None:
			try:
				nanlabel = self.sklabelencoder.inverse_transform(self.nan)
				res = np.array(res,dtype="float")
				res[res==nanlabel] = np.nan
			except:
				pass
		return res 
	def fit_transform(self,train):
		self.fit(train)
		return self.transform(train)


class ExOneHotEncoder(object):
    def __init__(self):
        self.skonehotencoder = OneHotEncoder(sparse=False)
    def _fit(self,train):
        self.containnan = train.isnull().sum()>0
        self.fillna = train.max()+1
        self.numcateg = train.nunique(dropna=False)
        cumindex=0
        self.todelete = []
        for i in range(len(self.containnan.values)):
            cumindex+=self.numcateg.values[i]
            if self.containnan.values[i] == True:
                self.todelete.append(cumindex-1)
    def fit(self,train):
        self._fit(train)
        self.skonehotencoder.fit(train.fillna(self.fillna).astype('int32'))
        
    def transform(self,test):
        encoded = self.skonehotencoder.transform(test.fillna(self.fillna).astype('int32'))
        encoded = np.delete(encoded,self.todelete,1)
        return encoded
    def fit_transform(self,train):
        self._fit(train)
        encoded = self.skonehotencoder.fit_transform(train.fillna(self.fillna).astype('int32'))
        encoded = np.delete(encoded,self.todelete,1)
        return encoded
    
def multifeat_onehot(train_nn):
    ohe = OneHotEncoder(sparse=False)
    train_nnmax = train_nn.max()
    tmpohe = ohe.fit_transform(train_nn.reshape((len(train_nn)*train_nn.shape[1],1)))[:,1:]\
    .reshape((len(train_nn),train_nn.shape[1],train_nnmax)).sum(axis=1).astype('int32')
    return tmpohe

    
        