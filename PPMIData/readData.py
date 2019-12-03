# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import os

data_file = 'final_PD_updrs2_updrs3_pdfeat_demo.csv'

TOTAL_VISITS = 17
TARGET_LABEL = 'AMBUL_SCORE'
time_from_bl_ix = 51

class PPMIData():
    '''
    Get PPMI data based on patient ids
    '''
    def __init__(self, pat_ids, data, batch_size, isNormal):
        self.batchSize = batch_size
        # get x and mask
        self.times = []
        self.x = [] # pat_id * timestep * features
        self.m = [] # pat_id * timestep * features
        self.y = []
        self.pat_ix = [] # pat ix to ids
        self.feat_dict = {} # faeture: ix
        self.isNormal = True
        
        features = data.columns.values.tolist()
        for i, feat in enumerate(features):
            if feat != "PATNO" and feat != "EVENT_ID":
                self.feat_dict[feat] = i-2
        
        print("\n feature vector len: " + str(len(self.feat_dict)))
        # mean and std for each feature
        self.mean = [0]*len(self.feat_dict)
        self.std = [0] * len(self.feat_dict)
        meancount = [0] * len(self.feat_dict)
        
        for pat_id in pat_ids:
            pat_data = data.loc[data['PATNO'] == pat_id, :]
            last_event = int(np.max(pat_data['EVENT_ID'].values))
            if last_event == 0:
                #print("Only one record found: " + str(pat_id))
                continue
            
            self.pat_ix.append(pat_id)            
            one_pat_data = []
            one_pat_mask = [] # matrix time * features
            labels = [] # not needed
            t_times = []
            for visit in range(last_event+1):
                pat_event_data = pat_data.loc[pat_data['EVENT_ID'] == visit, :].values
                
                
                pat_event_data = pat_event_data.tolist()
                mask = [0] * len(self.feat_dict)
                if len(pat_event_data) == 0:
                    pat_event_data = [-1] * len(self.feat_dict)
                    t_times.append(t_times[-1]+90)
                else:
                    pat_event_data = pat_event_data[0][2:]
                    t_times.append(pat_event_data[time_from_bl_ix]*365) # time from BL * 365
                        
                labels.append(pat_event_data[self.feat_dict[TARGET_LABEL]])
                for i, feat in enumerate(pat_event_data):
                    if feat == -1:
                        mask[i] = 0
                        pat_event_data[i] = 0
                    else:
                        mask[i] = 1
                        meancount[i] += 1
                        self.mean[i] += feat
                            
                one_pat_data.append(pat_event_data)
                one_pat_mask.append(mask)
            
            self.x.append(one_pat_data)
            self.m.append(one_pat_mask)
            self.y.append(labels)
            self.times.append(t_times)
            #self.times.append(list(range(last_event+1))) #TODO: change to times
            
        for i in range(len(self.mean)):
            self.mean[i] = self.mean[i]/meancount[i]
            
        for pat_ix in range(len(self.m)):
            one_pat_mask = self.m[pat_ix]
            for time_ix in range(len(one_pat_mask)):
                one_ts_feat = one_pat_mask[time_ix]
                for feat_ix, feat_mask in enumerate(one_ts_feat):
                    if feat_mask == 1:
                        feat = self.x[pat_ix][time_ix][feat_ix]
                        self.std[feat_ix] += (feat-self.mean[feat_ix])**2
        
        for i in range(len(self.std)):
            if meancount[i] == 1:
                self.std[i] = 0
            else:
                self.std[i] = math.sqrt(1/(meancount[i]-1) * self.std[i])
          
        if isNormal:
            self.normalize()
        
        self.construct_delta()
            
    '''
    Construct delta matrix
    '''
    def construct_delta(self):
        x_lengths=[] #
        deltaPre=[] #time difference 
        lastvalues=[] # if missing, last values
        deltaSub=[]
        subvalues=[]
        m = self.m
        for h in range(len(self.x)):
            # oneFile: steps*value_number
            oneFile=self.x[h]
            one_time=self.times[h]
            x_lengths.append(len(oneFile))
            
            one_deltaPre=[]
            one_lastvalues=[]
            
            one_deltaSub=[]
            one_subvalues=[]
            
            one_m=m[h]
            for i in range(len(oneFile)):
                t_deltaPre=[0.0]*len(oneFile[i])
                t_lastvalue=[0.0]*len(oneFile[i])
                one_deltaPre.append(t_deltaPre)
                one_lastvalues.append(t_lastvalue)
                
                if i==0:
                    for j in range(len(oneFile[i])):
                        one_lastvalues[i][j]=0.0 if one_m[i][j]==0 else oneFile[i][j]
                    continue
                for j in range(len(oneFile[i])):
                    if one_m[i-1][j]==1:
                        one_deltaPre[i][j]=one_time[i]-one_time[i-1]
                    if one_m[i-1][j]==0:
                        one_deltaPre[i][j]=one_time[i]-one_time[i-1]+one_deltaPre[i-1][j]
                        
                    if one_m[i][j]==1:
                        one_lastvalues[i][j]=oneFile[i][j]
                    if one_m[i][j]==0:
                        one_lastvalues[i][j]=one_lastvalues[i-1][j]
        
            for i in range(len(oneFile)):
                t_deltaSub=[0.0]*len(oneFile[i])
                t_subvalue=[0.0]*len(oneFile[i])
                one_deltaSub.append(t_deltaSub)
                one_subvalues.append(t_subvalue)
            #construct array 
            for i in range(len(oneFile)-1,-1,-1):    
                if i==len(oneFile)-1:
                    for j in range(len(oneFile[i])):
                        one_subvalues[i][j]=0.0 if one_m[i][j]==0 else oneFile[i][j]
                    continue
                for j in range(len(oneFile[i])):
                    if one_m[i+1][j]==1:
                        one_deltaSub[i][j]=one_time[i+1]-one_time[i]
                    if one_m[i+1][j]==0:
                        one_deltaSub[i][j]=one_time[i+1]-one_time[i]+one_deltaSub[i+1][j]
                        
                    if one_m[i][j]==1:
                        one_subvalues[i][j]=oneFile[i][j]
                    if one_m[i][j]==0:
                        one_subvalues[i][j]=one_subvalues[i+1][j]   
                
            
            #m.append(one_m)
            deltaPre.append(one_deltaPre)
            lastvalues.append(one_lastvalues)
            deltaSub.append(one_deltaSub)
            subvalues.append(one_subvalues)
            
        self.m=m
        self.deltaPre=deltaPre
        self.lastvalues=lastvalues
        self.deltaSub=deltaSub
        self.subvalues=subvalues
        self.x_lengths=x_lengths
        self.maxLength=max(x_lengths)
        
    '''
    Normalize the features 
    '''
    def normalize(self):
        for pat_ix in range(len(self.m)):
            one_pat_mask = self.m[pat_ix]
            for time_ix in range(len(one_pat_mask)):
                one_ts_feat = one_pat_mask[time_ix]
                for feat_ix, feat_mask in enumerate(one_ts_feat):
                    if feat_mask == 1:
                        feat_value = self.x[pat_ix][time_ix][feat_ix]
                        if self.std[feat_ix] != 0:
                            feat_value = (feat_value-self.mean[feat_ix])/self.std[feat_ix]
                        else:
                            feat_value = 0
                        
                        self.x[pat_ix][time_ix][feat_ix] = feat_value
                    
    '''
    Get next batch of data
    '''
    def nextBatch(self):
        i=1
        while i*self.batchSize<=len(self.x):
            x=[]
            y=[]
            m=[]
            deltaPre=[]
            x_lengths=[]
            lastvalues=[]
            deltaSub=[]
            subvalues=[]
            imputed_deltapre=[]
            imputed_m=[]
            imputed_deltasub=[]
            mean=self.mean
            files = [] # dummy
            for j in range((i-1)*self.batchSize,i*self.batchSize):
                x.append(self.x[j])
                #y.append(self.y[j])
                m.append(self.m[j])
                deltaPre.append(self.deltaPre[j])
                deltaSub.append(self.deltaSub[j])
                
                x_lengths.append(self.x_lengths[j])
                lastvalues.append(self.lastvalues[j])
                subvalues.append(self.subvalues[j])
                jj=j-(i-1)*self.batchSize
                #times.append(self.times[j])
                while len(x[jj])<self.maxLength:
                    t1=[0.0]*(len(self.feat_dict))
                    x[jj].append(t1)
                    #times[jj].append(0.0)
                    t2=[0]*(len(self.feat_dict))
                    m[jj].append(t2)
                    t3=[0.0]*(len(self.feat_dict))
                    deltaPre[jj].append(t3)
                    t4=[0.0]*(len(self.feat_dict))
                    lastvalues[jj].append(t4)
                    t5=[0.0]*(len(self.feat_dict))
                    deltaSub[jj].append(t5)
                    t6=[0.0]*(len(self.feat_dict))
                    subvalues[jj].append(t6)
            for j in range((i-1)*self.batchSize,i*self.batchSize):
                one_imputed_deltapre=[]
                one_imputed_deltasub=[]
                one_G_m=[]
                for h in range(0,self.x_lengths[j]):
                    if h==0:
                        one_f_time=[0.0]*(len(self.feat_dict))
                        one_imputed_deltapre.append(one_f_time)
                        try:
                            one_sub=[self.times[j][h+1]-self.times[j][h]]*\
                            (len(self.feat_dict))
                        except:
                            print("error: "+str(h)+" "+str(len(self.times[j])))
                        one_imputed_deltasub.append(one_sub)
                        one_f_g_m=[1.0]*(len(self.feat_dict))
                        one_G_m.append(one_f_g_m)
                    elif h==self.x_lengths[j]-1:
                        one_f_time=[self.times[j][h]-self.times[j][h-1]]*\
                            (len(self.feat_dict))
                        one_imputed_deltapre.append(one_f_time)
                        one_sub=[0.0]*(len(self.feat_dict))
                        one_imputed_deltasub.append(one_sub)
                        one_f_g_m=[1.0]*(len(self.feat_dict))
                        one_G_m.append(one_f_g_m)
                    else:
                        one_f_time=[self.times[j][h]-self.times[j][h-1]]*\
                                (len(self.feat_dict))
                        one_imputed_deltapre.append(one_f_time)
                        one_sub=[self.times[j][h+1]-self.times[j][h]]*\
                                (len(self.feat_dict))
                        one_imputed_deltasub.append(one_sub)
                        one_f_g_m=[1.0]*(len(self.feat_dict))
                        one_G_m.append(one_f_g_m)
                while len(one_imputed_deltapre)<self.maxLength:
                    one_f_time=[0.0]*(len(self.feat_dict))
                    one_imputed_deltapre.append(one_f_time)
                    one_sub=[0.0]*(len(self.feat_dict))
                    one_imputed_deltasub.append(one_sub)
                    one_f_g_m=[0.0]*(len(self.feat_dict))
                    one_G_m.append(one_f_g_m)
                imputed_deltapre.append(one_imputed_deltapre)
                imputed_deltasub.append(one_imputed_deltasub)
                imputed_m.append(one_G_m)
                
            i+=1
            if self.isNormal:
                yield  x,y,[0.0]*(len(self.feat_dict)),m,deltaPre,x_lengths,lastvalues,\
                files,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub
            else:
                yield  x,y,mean,m,deltaPre,x_lengths,lastvalues,files,imputed_deltapre,\
                imputed_m,deltaSub,subvalues,imputed_deltasub
                
    def shuffle(self, batch_size, isTrain):
        pass
    
    
class ReadPPMIData():
    '''
    Create train and test data splits
    '''
    def __init__(self, train_test_split=0.9, batch_size=16, data_path='./', isNormal=False):
        df = pd.read_csv(os.path.join(data_path, data_file))
        pat_ids = df['PATNO'].unique()
        ix = np.arange(len(pat_ids)); np.random.shuffle(ix)
        train_ix = ix[0:int(len(ix) * train_test_split)]
        test_ix = ix[int(len(ix) * train_test_split):]
        train_pat_ids = pat_ids[train_ix]; test_pat_ids = pat_ids[test_ix]
        self.train_pat_ids = train_pat_ids
        self.test_pat_ids = test_pat_ids
        self.whole_data = df
        self.batch_size = batch_size
        self.isNormal = isNormal
        
    def read_train(self):
        train_data = PPMIData(self.train_pat_ids, self.whole_data, self.batch_size, self.isNormal)
        self.maxLength = train_data.maxLength
        return train_data
        
    def read_test(self):
        test_data = PPMIData(self.test_pat_ids, self.whole_data, self.batch_size, self.isNormal)
        test_data.maxLength = max(test_data.maxLength, self.maxLength)
        return test_data
 
def get_shapes(dataset):
    def getsh(d, l):
        if type(d) == list:
            l.append(len(d))
            if len(d) != 0:
                getsh(d[0], l)
            
    for batch in dataset.nextBatch():
        for vec in batch:
            shape = []
            getsh(vec, shape)
            print("shape : " + str(shape))
        
        break  
     
if __name__ == "__main__":
    rp = ReadPPMIData()
    train_data = rp.read_train()
    get_shapes(train_data)        
    test_data = rp.read_test()
    get_shapes(test_data)