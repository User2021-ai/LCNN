#!/usr/bin/env python
# coding: utf-8

# In[25]:


import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt 
from random import uniform, seed
import numpy as np 
import time
from operator import itemgetter # for sort list of list
from math import e
import math
import copy
import random
import os 
import warnings
warnings.filterwarnings('ignore')
 
from skimage.io import imread
from matplotlib import pyplot as plt
 
import keras
from keras import optimizers
from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation, concatenate
from keras.layers import ELU, PReLU, LeakyReLU

from keras.models import Model
import networkx as nx
 

from scipy import stats
from ast import literal_eval
from keras.models import load_model
from keras import models

from decimal import Decimal
import tensorflow as tf    


# # Metrics based on node degree

# In[26]:


def dic_D_1_weights_all_nodes(G):    
   dic_D_1_Nodes={}
   for u in G: 
       dic_D_1_Nodes[u]=nx.degree(G,u) 
   return dic_D_1_Nodes


def dic_D_2_weights_all_nodes(G,dic_D_1_Nodes):    
    
   dic_D_2_Nodes={}
   for u in G:
       Tv=[n for n in G.neighbors(u)] # neighbors of v
       D2=dic_D_1_Nodes[(u)]
       for v in Tv:
           D2+=dic_D_1_Nodes[(v)]
       dic_D_2_Nodes[u]=D2        
   return dic_D_2_Nodes


def dic_D_3_weights_all_nodes(G,dic_D_2_Nodes):        
   dic_D_3_Nodes={}
   for u in G:
       Tv=[n for n in G.neighbors(u)] # neighbors of v
       D3=dic_D_2_Nodes[(u)]
       for v in Tv:
           D3+=dic_D_2_Nodes[(v)]
       dic_D_3_Nodes[u]=D3       
   return dic_D_3_Nodes


# # Metrics based on H-index

# In[27]:



def H_index(G,node):
   Tv=[n for n in G.neighbors(node)] # neighbors of v.
   # sorting in ascending order
   citations=[nx.degree(G,v) for v in Tv ]
   citations.sort()
     
   # iterating over the list
   for i, cited in enumerate(citations):
         
       # finding current result
       result = len(citations) - i          
       # if result is less than or equal
       # to cited then return result
       if result <= cited:
           return result           
   return 0

def H_index_weights_of_All_nodes(G):
    h_index_1_Nodes={} 
    h_index_2_Nodes={}
    h_index_3_Nodes={} 
    
    for u in G:
        H=H_index(G,u)
        h_index_1_Nodes[u]=H 

    
    for u in G:
       
        Tv=[n for n in G.neighbors(u)] # neighbors of v.
        h_index_2=h_index_1_Nodes[(u)]
        for n in Tv:
            h_index_2+=h_index_1_Nodes[(n)]
        h_index_2_Nodes[u]=h_index_2
        
    for u in G:        
        Tv=[n for n in G.neighbors(u)] # neighbors of v.
        h_index_3=h_index_2_Nodes[(u)]
        for n in Tv:
            h_index_3+=h_index_2_Nodes[(n)]
        h_index_3_Nodes[u]=h_index_3    
        
    return h_index_1_Nodes,h_index_2_Nodes,h_index_3_Nodes


# # Structural channel sets of node representations

# In[28]:


def metrics_one_two_hop_Adj_mat_of_node(G,L,node,dic_D1,dic_D2,dic_D3,dic_H1,dic_H2,dic_H3 ):
      
    one_hop = list(G.adj[node])  
    one_hop_weight_a ={}  
    for u in one_hop:
        one_hop_weight_a[u]=dic_D3[(u)]   
    
    sorted_list=sorted(one_hop_weight_a.items(),key=lambda x:x[1],reverse=True)   
     
        
    selected_nei = [node]+[i for i,j in sorted_list[:L] ]  
    # padding
    if len(one_hop) < L:  
        selected_nei=selected_nei+[-1 for i in range(L-len(one_hop))]      
    # extract ADJ Mat
    arr_D1=[]   
    arr_D2=[]    
    arr_D3=[] 
    
    
    arr_H1=[] 
    arr_H2=[]
    arr_H3=[] 
      
    i_index=0 
    for i in selected_nei:
        col_D1 = []
        col_D2 = []                
        col_D3 = []
       
        
        col_H1 = [] 
        col_H2 = []
        col_H3 = [] 
        
        j_index=0
        for j in selected_nei:
            if i_index==j_index:
                col_D1.append(dic_D1[(node)])
                col_D2.append(dic_D2[(node)])                                              
                col_D3.append(dic_D3[(node)])
                
                
                col_H1.append(dic_H1[(node)])
                col_H2.append(dic_H2[(node)])
                col_H3.append(dic_H3[(node)])
                
                 
                
            else:
                if G.has_edge(i,j):
                    if(i_index==0 and i_index<j_index):
                        col_D1.append(dic_D1[(j)])
                        col_D2.append(dic_D2[(j)])                      
                        col_D3.append(dic_D3[(j)])
                        
                        col_H1.append(dic_H1[(j)])
                        col_H2.append(dic_H2[(j)])
                        col_H3.append(dic_H3[(j)])
                
                    elif(j_index==0 and i_index>j_index):
                        col_D1.append(dic_D1[(i)])
                        col_D2.append(dic_D2[(i)])                       
                        col_D3.append(dic_D3[(i)])
                       
                        
                        col_H1.append(dic_H1[(i)])
                        col_H2.append(dic_H2[(i)])
                        col_H3.append(dic_H3[(i)])
                       
                    else:
                        col_D1.append(1)
                        col_D2.append(1)                     
                        col_D3.append(1)                       
                        
                        col_H1.append(1)
                        col_H2.append(1)
                        col_H3.append(1)
                        
                else:
                    col_D1.append(0)
                    col_D2.append(0)
                    col_D3.append(0)                   
                    
                    col_H1.append(0)
                    col_H2.append(0)
                    col_H3.append(0)
                    
                    
            j_index+=1
        i_index+=1
        arr_D1.append(col_D1) 
        arr_D2.append(col_D2)         
        arr_D3.append(col_D3)        
        
        arr_H1.append(col_H1)
        arr_H2.append(col_H2)
        arr_H3.append(col_H3)
        
    return np.array(arr_D1),np.array(arr_D2),np.array(arr_D3),np.array(arr_H1),np.array(arr_H2),np.array(arr_H3)



def metrics_one__hop_Adj_mat_of_all_nodes(G,L  ):  
    dic_local_embedding={}
    dic_semi_embedding={}
    
    dic_D1=dic_D_1_weights_all_nodes(G)
    dic_D2=dic_D_2_weights_all_nodes(G,dic_D1)    
    dic_D3=dic_D_2_weights_all_nodes(G,dic_D2)  
    
    
    dic_H1,dic_H2,dic_H3=H_index_weights_of_All_nodes(G)
        
    for node in G:
        hop_1_arr_D1,hop_1_arr_D2,hop_1_arr_D3,hop_1_arr_H1,hop_1_arr_H2,hop_1_arr_H3=metrics_one_two_hop_Adj_mat_of_node(G,L,node,dic_D1,dic_D2, dic_D3,dic_H1,dic_H2,dic_H3 )
 
        dic_local_embedding[node]=hop_1_arr_D1,hop_1_arr_D2, hop_1_arr_D3    
        dic_semi_embedding[node]= hop_1_arr_H1,hop_1_arr_H2,hop_1_arr_H3 
         
    return dic_local_embedding,dic_semi_embedding


# In[29]:


def create_LCNN_Model(path_Train_Data,path_SIR_Train_Data,L,Kernel_size,MaxPooling,  dense,learning_rate,epochN):
    
    in_channel_L=3
    in_channel_S=3 

    data_G = loadData(path_Train_Data)
    data_G_sir = pd.read_csv(path_SIR_Train_Data)    
     
 
    data_G_label = dict(zip(np.array(data_G_sir['Node'],dtype=str),data_G_sir['SIR']))    
    
    dic_local_embedding,dic_semi_embedding=metrics_one__hop_Adj_mat_of_all_nodes(data_G,L  )
     
    x1_train = []
    x2_train = []
      
    y_train = []
    
    for node in data_G:
        x1_train.append(dic_local_embedding[(node)])
        x2_train.append(dic_semi_embedding[(node)])
        
        y_train.append(data_G_label[(node)])
      
    x1_train=np.array(x1_train)
    x2_train=np.array(x2_train)
         
    x1_train=x1_train.reshape(-1, L+1, L+1,  in_channel_L)
    x2_train=x2_train.reshape(-1, L+1, L+1, in_channel_S)
         
    print(x1_train.shape)
    print(x2_train.shape)
    
    y_train = np.array(y_train)
    
    input_shape_L = (L+1, L+1, in_channel_L)
    input_shape_S = (L+1, L+1, in_channel_S)
        
    def create_convolution_layers(input_model,input_shape):
        
        model = Conv2D(filters=18,kernel_size= Kernel_size,strides=1, padding='same', input_shape=input_shape)(input_model)       
        model = keras.layers.BatchNormalization()(model)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling2D((MaxPooling, MaxPooling),padding='same')(model)
               
        
        model = Conv2D(filters=48,kernel_size= Kernel_size,strides=1, padding='same')(model)
        model = keras.layers.BatchNormalization()(model)
        model = LeakyReLU(alpha=0.1)(model)
        model = MaxPooling2D((MaxPooling, MaxPooling),padding='same')(model)
         
        
        return model
      
    
    local_input = Input(shape=input_shape_L)
    # local_input.shape
    conv_1 = create_convolution_layers(local_input,input_shape_L)
    conv_1 = keras.layers.TimeDistributed(Flatten())(conv_1) 
    conv_1=keras.layers.GlobalAveragePooling1D()(conv_1) 
        
    semi_input = Input(shape=input_shape_S)
    conv_2 = create_convolution_layers(semi_input,input_shape_S)
    conv_2 = keras.layers.TimeDistributed(Flatten())(conv_2)
    conv_2=keras.layers.GlobalAveragePooling1D()(conv_2)
 
     
    convAall = concatenate([conv_1,conv_2])
    
    dense = Dense(dense)(convAall) 
    dense=LeakyReLU(alpha=0.1)(dense)
    dense = Dense(1)(dense)
    output=LeakyReLU(alpha=0.1)(dense)

     
    model = Model(inputs=[local_input, semi_input], outputs=[output])
    opt= keras.optimizers.Adam(learning_rate=learning_rate)  
    model.compile(loss="mse", optimizer=opt)
    model.summary()
    
    history = model.fit([x1_train,x2_train],y_train,epochs =epochN,shuffle=True,batch_size=4)
    
    
    loss = history.history['loss']
    epochs_range = range(epochN)
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 1, 1)  
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.legend(loc='upper right')
    plt.title('Training Loss')
    plt.show()
    return model, history


# In[33]:



def loadData(nameDataset,sep=","):    
   df= pd.read_csv(nameDataset, sep=sep,names=['FromNodeId','ToNodeId'])       
   
   G = nx.from_pandas_edgelist(df, source="FromNodeId", target="ToNodeId")    
   G.remove_edges_from(nx.selfloop_edges(G))
   
   G.remove_nodes_from(['FromNodeId', 'ToNodeId',])
   #print(nx.info(G))
   return G


def get_data_to_model(G,L ):
   dic_local_embedding,dic_semi_embedding=metrics_one__hop_Adj_mat_of_all_nodes(G,L )

   x1_train = []
   x2_train = []
  
   in_channel_L=3
   in_channel_S=3
   
   for node in G:
       x1_train.append(dic_local_embedding[(node)])
       x2_train.append(dic_semi_embedding[(node)])
       
   x1_train=np.array(x1_train)
   x2_train=np.array(x2_train)
   
   x1_train=x1_train.reshape(-1, L+1, L+1, in_channel_L)
   x2_train=x2_train.reshape(-1, L+1, L+1, in_channel_S)
   
   # print(x1_train.shape)
   # print(x2_train.shape)  
   return x1_train,x2_train    

def get_sir_list(pathDataset,nameDataset,sir_rang_list):
   sir_list=[]   
   for a_tau in sir_rang_list:
       sir = pd.read_csv(pathDataset+nameDataset+'/'+nameDataset+'_a['+str(round(a_tau,1))+']_.csv')
      
       sir_list.append(dict(zip(np.array(sir['Node'],dtype=str),sir['SIR'])))
   return sir_list

def nodesRank(rank):
   SR = sorted(rank)
   re = []
   for i in SR:
       re.append(rank.index(i))
   return re

def get_algo_list(pathDataset,dataName,algoName):
   algo_list=[]
   df = pd.read_csv(pathDataset)     
   df=df[df['Dataset']==dataName]
   df=df[df['Algo']==algoName]    
   algo_list=literal_eval(df['Seed'].iloc[0])
   algo_list=algo_list         
   return algo_list
              
def compare_tau(sir_list,alg_list):   
   alg_tau_list=[]   
   for sir in sir_list:        
       sir_sort = [i for i,j in sorted(sir.items(),key=lambda x:x[1],reverse=True)]     
       tau3,_ = stats.kendalltau(nodesRank(alg_list),nodesRank(sir_sort))            
       alg_tau_list.append(tau3)        
   return alg_tau_list    


def rank_dataset_using_LCNN(model,model_name,input_Datasets_to_pred,path_input_Datasets,path_SIR_input_Datasets,
                           sir_rang_list,path_saved_ranked_node,L,  
                            name_Train_Data,sir_a_value_Train_Data ,Kernel_size,MaxPooling,  Dense,learning_rate):
   df_seed_LCNN = pd.DataFrame( columns=['Dataset','Algo','Seed','time'])
   df_tau_result = pd.DataFrame( columns=['Dataset', 'sir_a_value_Train_Data',  
                                          'Algo','Tau','Dense','MaxPooling','Kernel_size','learning_rate'])

   for dataName in input_Datasets_to_pred:
       start_time = time.time()   
       G = loadData(path_input_Datasets+dataName+'.csv')
       x1_train,x2_train=get_data_to_model(G,L  )
       data_predictions = model.predict([x1_train,x2_train])
       nodes = list(G.nodes())
       my_pred = [i for i,j in sorted(dict(zip(nodes,data_predictions)).items(),key=lambda x:x[1],reverse=True)] 
       timelapse=(time.time() - start_time)   
       #df2 = {'Dataset': dataName, 'Algo': model_name, 'Seed': my_pred,'time':timelapse}
       #df_seed_LCNN=df_seed_LCNN.append(df2, ignore_index = True)
       
       print('-------------------------------------------------------------')
       print('done', model_name,' in  ', dataName)
       print('-------------------------------------------------------------')
       G_SIR = get_sir_list(path_SIR_input_Datasets,dataName,sir_rang_list)        
       tau=compare_tau(G_SIR,my_pred)
       print('tau=',tau)
       df3 = {'Dataset': dataName, 
              'sir_a_value_Train_Data':sir_a_value_Train_Data,   'Algo': model_name, 'Tau': tau,
             'Dense':Dense,'MaxPooling':MaxPooling,'Kernel_size':Kernel_size,'learning_rate':learning_rate}
       df_tau_result=df_tau_result.append(df3, ignore_index = True)
       
       
       #df_seed_LCNN.to_csv(model_name+'__Seed.csv')
       df_tau_result.to_csv(path_saved_ranked_node+'/'+model_name+'__Tau.csv')
   
 
 


# In[34]:


def reset_random_seeds():
    os.environ['PYTHONHASHSEED']=str(2)
    tf.random.set_seed(2)
    np.random.seed(2)
    random.seed(2)
reset_random_seeds()

L=40
epochN=200
dense=1024
learning_ra=0.0005  
MaxPooling=2 
Kernel_size=2 

name_Train_Data='BA-1k.csv' 
sir_a_value_Train_Data='1.5' 
path_saved_ranked_node='Results'
sir_rang_list = np.arange(1.0,2.0,0.1)


input_Datasets_to_pred=['BA-8k','Gnutella' ]
sir_rang_list = np.arange(1.0,2.0,0.1)
 



path_Train_Data='Data/'+name_Train_Data
path_SIR_Train_Data='SIR/'+name_Train_Data[:-4]+'/'+name_Train_Data[:-4]+'_a['+sir_a_value_Train_Data+']_.csv'

         
model_name='LCNN'+'_Ker_'+str(Kernel_size)+'_Max_'+str(MaxPooling)+'_dense_'+str(dense)+'lear_'+str(learning_ra) 
PATH_saved_model = "Models/"+model_name+".h5"

#model,_=create_LCNN_Model(path_Train_Data,path_SIR_Train_Data,L,Kernel_size,MaxPooling, dense,learning_ra,epochN  )
#model.save(PATH_saved_model) 

model=models.load_model(PATH_saved_model, compile=False)
path_input_Datasets='Data/'
path_SIR_input_Datasets='SIR/'

rank_dataset_using_LCNN(model,model_name,input_Datasets_to_pred,path_input_Datasets,path_SIR_input_Datasets,
                        sir_rang_list,path_saved_ranked_node,L,
                        name_Train_Data,sir_a_value_Train_Data ,Kernel_size,MaxPooling,  dense,learning_ra )


# In[ ]:




