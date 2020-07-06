import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import tensorflow_text
import os
from langdetect import detect
import sys
inputfile = sys.argv[2]
outputfile = sys.argv[4]

import pandas as pd
import xml.etree.ElementTree as et


def parse_XML(df_cols):
    path = str(inputfile)+'/en/'
    rows = []
    ids = []
    langs = []
    out_df=pd.DataFrame()
    for filename in os.listdir(path):
        if not filename.endswith('.xml'): continue
        xml_file = os.path.join(path, filename)
        xtree = et.parse(xml_file)
        xroot = xtree.getroot() 
        xchild=xroot.getchildren()
        for node in xchild:
            res = []
            qs=(node.findall("document"))
            for q in qs:
                res.append(q.text)
            rows.append({i: res[i] for i, _ in enumerate(df_cols)})   
                        
            
            ids.append(filename[:-4])
            langs.append(detect(qs[0].text))
            out_df = pd.DataFrame(rows)
            out_df['id']=ids
            out_df['lang']=langs
            #out_df_id=pd.DataFrame(ids,columns=['id'])
            #out_df_lang=pd.DataFrame(langs,columns=['lang'])
    #result=pd.concat([out_df_id,out_df_lang],axis=1)
    #new_out_df=pd.concat([out_df,result],axis=1)
    return out_df

df1=parse_XML(["document","document","document","document","document","document","document","document","document","document",
"document","document","document","document","document","document","document","document","document","document",
"document","document","document","document","document","document","document","document","document","document",
"document","document","document","document","document","document","document","document","document","document",
"document","document","document","document","document","document","document","document","document","document",
"document","document","document","document","document","document","document","document","document","document",
"document","document","document","document","document","document","document","document","document","document",
"document","document","document","document","document","document","document","document","document","document",
"document","document","document","document","document","document","document","document","document","document",
"document","document","document","document","document","document","document","document","document","document"])  
def parse_XML_spanish(df_cols):
    path = str(inputfile)+'/es/'
    rows = []
    ids = []
    langs = []
    out_df=pd.DataFrame()
    for filename in os.listdir(path):
        if not filename.endswith('.xml'): continue
        xml_file = os.path.join(path, filename)
        xtree = et.parse(xml_file)
        xroot = xtree.getroot() 
        xchild=xroot.getchildren()
        for node in xchild:
            res = []
            qs=(node.findall("document"))
            for q in qs:
                res.append(q.text)
            rows.append({i: res[i] for i, _ in enumerate(df_cols)})   
                        
            
            ids.append(filename[:-4])
            langs.append(detect(qs[0].text))
            out_df = pd.DataFrame(rows)
            out_df['id']=ids
            out_df['lang']=langs
            #out_df_id=pd.DataFrame(ids,columns=['id'])
            #out_df_lang=pd.DataFrame(langs,columns=['lang'])
    #result=pd.concat([out_df_id,out_df_lang],axis=1)
    #new_out_df=pd.concat([out_df,result],axis=1)
    return out_df 
	
df2=parse_XML_spanish(["document","document","document","document","document","document","document","document","document","document",
"document","document","document","document","document","document","document","document","document","document",
"document","document","document","document","document","document","document","document","document","document",
"document","document","document","document","document","document","document","document","document","document",
"document","document","document","document","document","document","document","document","document","document",
"document","document","document","document","document","document","document","document","document","document",
"document","document","document","document","document","document","document","document","document","document",
"document","document","document","document","document","document","document","document","document","document",
"document","document","document","document","document","document","document","document","document","document",
"document","document","document","document","document","document","document","document","document","document"])  	
new_out_df=pd.concat([df1,df2],axis=0,ignore_index=True)	

#module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'
#module_url='https://tfhub.dev/google/universal-sentence-encoder-large/5'
embed = hub.load('/home/majumder20a/Desktop/software/large_3/')
no_author=len(new_out_df)
submission=pd.DataFrame()
new_result=np.zeros((no_author,100,512))
submission['id']=new_out_df['id']
submission['lang']=new_out_df['lang']
for col in new_out_df:
    if col == 'id'or col=='lang':
        continue
    for i, row_value in new_out_df[col].iteritems():
        new_out_df[col][i] = embed([row_value])

result_new=new_out_df.iloc[:,0:100]#iloc takes integer for column slicing ,loc takes index of columns for slicing
result_new=np.asarray(result_new)
for i in range(0,no_author):
    for j in range(0,100):
        for k in range(512):
            new_result[i,j,k]=result_new[i][j][0][k]

from keras.models import load_model,Sequential
from keras.layers import Layer,LSTM,Dense,Bidirectional
import keras.backend as K

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()



model1 = Sequential()
#model.add(Input(shape=(100, 1),dtype=object))
model1.add(LSTM(256,return_sequences=True,input_shape=(100,512),dropout=0.3,recurrent_dropout=0.2))
#model.add(TimeDistributed(Dense(128,activation='relu')))
#model.add(Bidirectional(LSTM(128,return_sequences=True)))
model1.add(attention())
model1.add(Dense(1,activation='sigmoid',trainable=True))
model1.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

model2 = Sequential()
#model.add(Input(shape=(100, 1),dtype=object))
model2.add(LSTM(256,return_sequences=True,input_shape=(100,512),dropout=0.3,recurrent_dropout=0.2))
#model.add(TimeDistributed(Dense(128,activation='relu')))
#model.add(Bidirectional(LSTM(128,return_sequences=True)))
model2.add(attention())
model2.add(Dense(1,activation='sigmoid',trainable=True))
model2.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

model1.load_weights('/home/majumder20a/Desktop/software/model_english.h5')
model2.load_weights('/home/majumder20a/Desktop/software/model_spanish.h5')
for i in range(len(submission)) : 
    if(submission.loc[i,'lang']=="en"):
        test_pred = model1.predict(new_result)
    if(submission.loc[i,'lang']=="es"):
        test_pred = model1.predict(new_result)
    submission['target'] = test_pred.round().astype(int)

from xml.etree.ElementTree import ElementTree
for i in range(len(submission)) : 
    #self closing xml element creation
    #parent_dir = outputfile
    parent_dir=str(outputfile)
    directory = submission.iloc[i,1]
    path_name = os.path.join(parent_dir, directory) 
    if(os.path.isdir('path_name')==False):
        os.makedirs(path_name,exist_ok=True)
    author=et.Element('author',{"id":submission.iloc[i,0],"lang":submission.iloc[i,1],"type":str(submission.iloc[i,2])})
    #author=et.Element('author',{'x':'y','a':'n'})
    #mydata=et.tostring(author)
    #mydata=ElementTree(author).write(myfile,method='xml')
    filename=submission.iloc[i,0]
    #myfile=open(str(filename)+".xml","w")
    #myfile.write(str(mydata))
    ElementTree(author).write(str(path_name)+"/"+str(filename)+".xml",method='xml')

 












