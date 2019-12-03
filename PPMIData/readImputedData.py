# -*- coding: utf-8 -*-

import os
import random
import csv

class ReadImputedPhysionetData:
    def __init__(self, dataPath ):
        #一个文件一个batch，但需要注意，x,y,delta之间的匹配
        #例子： batch1y,batch1x,batch1delta
        #batchid从1开始
        self.files = os.listdir(dataPath)
        self.dataPath=dataPath
        self.count=int(len(self.files)/3)
        
    def load(self):
        count=int(self.count)
        self.x=[]
        self.y=[]
        self.delta=[]
        self.x_lengths=[]
        self.m=[]
        for i in range(1,count+1):
            file_x=open(os.path.join(self.dataPath,"batch"+str(i)+"x"))
            this_x,this_lengths=self.readx(file_x)
            self.x.extend(this_x)
            self.x_lengths.extend(this_lengths)
            file_x.close()
            '''
            file_y=open(os.path.join(self.dataPath,"batch"+str(i)+"y"))
            file_delta=open(os.path.join(self.dataPath,"batch"+str(i)+"delta"))
            self.y.extend(self.ready(file_y))
            this_delta,this_m=self.readdelta(file_delta)
            self.delta.extend(this_delta)
            self.m.extend(this_m)
            file_y.close()
            file_delta.close()
            '''
        self.maxLength=len(self.x[0])
        
    def readx(self,x):
        this_x=[]
        this_lengths=[]
        count=1
        for line in x.readlines():
            if count==1:
                words=line.strip().split(",")
                for w in words:
                    if w=='':
                        continue
                    this_lengths.append(int(w))
            else:
                if "end" in line:
                    continue
                if "begin" in line:
                    d=[]
                    this_x.append(d)
                else:
                    words=line.strip().split(",")
                    oneclass=[]
                    for w in words:
                        if w=='':
                            continue
                        oneclass.append(float(w))
                    this_x[-1].append(oneclass)
            count+=1
        return this_x,this_lengths
    
if __name__ == "__main__":
    train = ReadImputedPhysionetData('../Gan_Imputation/imputation_train_results/WGAN_no_mask/30_8_16_64_0.001_100_True_True_False_0.1_0.5/')
    test = ReadImputedPhysionetData('../Gan_Imputation/imputation_test_results/WGAN_no_mask/30_8_16_64_0.001_100_True_True_False_0.1_0.5/')
    train.load(); test.load()
    all_data = []
    all_data.extend(train.x)
    all_data.extend(test.x)
    final_results = 'final_imputed_data.csv'
    columns = ["PATNO","EVENT_ID","NUPSOURC","NP2SPCH","NP2SALV","NP2SWAL","NP2EAT","NP2DRES","NP2HYGN","NP2HWRT","NP2HOBB","NP2TURN","NP2TRMR","NP2RISE","NP2WALK","NP2FREZ","NP3SPCH","NP3FACXP","NP3RIGN","NP3RIGRU","NP3RIGLU","PN3RIGRL","NP3RIGLL","NP3FTAPR","NP3FTAPL","NP3HMOVR","NP3HMOVL","NP3PRSPR","NP3PRSPL","NP3TTAPR","NP3TTAPL","NP3LGAGR","NP3LGAGL","NP3RISNG","NP3GAIT","NP3FRZGT","NP3PSTBL","NP3POSTR","NP3BRADY","NP3PTRMR","NP3PTRML","NP3KTRMR","NP3KTRML","NP3RTARU","NP3RTALU","NP3RTARL","NP3RTALL","NP3RTALJ","NP3RTCON","NHY","ANNUAL_TIME_BTW_DOSE_NUPDRS","CURRENT_APPRDX","AGE","TIME_FROM_BL","TIME_SINCE_DIAGNOSIS","TIME_SINCE_FIRST_SYMPTOM","TOTAL_UPDRS2","TOTAL_UPDRS3","TOTAL_UPDRS2_3","AMBUL_SCORE","DYSKPRES_0","DYSKPRES_1","DYSKIRAT_0","DYSKIRAT_1","ON_OFF_DOSE_0.0","ON_OFF_DOSE_1.0","PD_MED_USE_0.0","PD_MED_USE_1.0","PD_MED_USE_2.0","PD_MED_USE_3.0","PD_MED_USE_4.0","PD_MED_USE_5.0","PD_MED_USE_6.0","PD_MED_USE_7.0","IS_TREATED_0.0","IS_TREATED_1.0","PDDXEST_1","PDDXEST_3","PDDXEST_ACT","PDDXEST_DAY","PDDXEST_MD","PDDXEST_MON","DXTREMOR_0","DXTREMOR_1","DXTREMOR_U","DXRIGID_0","DXRIGID_1","DXRIGID_U","DXBRADY_0","DXBRADY_1","DXBRADY_U","DXPOSINS_0","DXPOSINS_1","DXPOSINS_U","DXOTHSX_0","DXOTHSX_1","DXOTHSX_U","DOMSIDE_1","DOMSIDE_2","DOMSIDE_3","APPRDX_1","APPRDX_2","APPRDX_3","APPRDX_4","APPRDX_5","APPRDX_6","APPRDX_7","APPRDX_8","P3GRP_1","P3GRP_2","GENDER_0","GENDER_1","GENDER_2","HISPLAT_0","HISPLAT_1","HISPLAT_2","RAINDALS_0","RAINDALS_1","RAINDALS_2","RAASIAN_0","RAASIAN_1","RAASIAN_2","RABLACK_0","RABLACK_1","RABLACK_2","RAHAWOPI_0","RAHAWOPI_1","RAHAWOPI_2","RAWHITE_0","RAWHITE_1","RAWHITE_2","RANOS_0","RANOS_1","RANOS_2","ENROLL_CAT_PD","HAS_PD_1.0"]
    with open(final_results, 'w') as fd:
        out = csv.writer(fd, quoting=csv.QUOTE_ALL)
        out.writerow(columns)
        for pat_record in all_data:
            for time_series in pat_record:
                out.writerow(time_series)