#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary Libraries

# In[3]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from scipy.stats import boxcox
import category_encoders as ce
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgbm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from statsmodels.api import Logit
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from flask import Flask, request, jsonify,render_template
import pickle


# # Importing Data

# In[4]:


df=pd.read_csv('diabetic_data_duplicate.csv')


# In[5]:


df.head()


# ### Since our primary objective is to predict the early re-admission of the patient, we will convert multi-class classification into binary classification.
# 
# 1. Class 0 - For no readmission and readmissions greater than 30 days
# 2. Class 1 - For readmissions within 30 days.

# In[6]:


df['readmitted'].replace(to_replace='>30',value='NO',inplace=True)


# # Data Preparation
# 
# #### There are few missing or incorrect values, let us treat them before finding patterns in the data and building model.
# 
# ## Race

# #### There are 2273 records where 'race' is missing.

# #### This is how we handle missing values of 'race':
# 1. Most of the patients from Caucasian race are from age between [70-80)
# 2. Most of the patients from African American race are from age between [50-60) and [60-70).
# 
# 1. By comparing age group, we can try to classify patient to respective race.
#     If the age range of patient is [70-80), patient can be classified to Caucasian race
#     If the age range of patient is [50-60) and [60-70), patient can be classified to AfricanAmerican race
#     Other unknown values can be classified as 'other' category

# In[7]:


df.loc[((df['race']=='?') & (df['age']=='[70-80)')),['race']]='Caucasian'
df.loc[((df['race']=='?') & ((df['age']=='[70-80)') | (df['age']=='[60-70)'))),['race']]='AfricanAmerican'


# In[8]:


df['race'].replace('?','Other',inplace=True)


# # Gender column

# In[9]:


df[df['gender']=='Unknown/Invalid']


# #### There are 3 records where gender value is 'unknown/Invalid'

# In[10]:


df[df['age']=='[70-80)'].groupby('gender')['age'].value_counts()


# In[11]:


df[df['age']=='[60-70)'].groupby('gender')['age'].value_counts()


# #### Two of them are in the age range of (70-80) and one is in the range of (60-70). We can see that, most of the patients in the age range of (70-80) are females and that of age range of (60-70) are males. We will impute, two of the values as female and one value as male

# In[12]:


def a(x):
    if((x['gender']=='Unknown/Invalid') & (x['age']=='[70-80)')):
        return 'Female'
    elif((x['gender']=='Unknown/Invalid') & (x['age']=='[60-70)')):
        return 'Male'
    else:
        return x['gender']


# In[13]:


df['gender']=df.apply(a,axis=1)


# # Age

# #### In the dataset, there are 10 bins as of now, but few bins have very less number of records. Hence its better to combine the age bins. We have re-divided the bins into five categories (0-30),(30-40),(40-50),(50-60),(60-70),(70-80),(80-100)

# In[14]:


def age_bins(x):
    if((x['age']=='[0-10)') | (x['age']=='[10-20)') | (x['age']=='[20-30)')):
        return '[0-30)'
    elif((x['age']=='[30-40)')):
        return'[30-40)'
    elif((x['age']=='[40-50)')):
        return'[40-50)'
    elif((x['age']=='[50-60)')):
        return '[50-60)'
    elif((x['age']=='[60-70)')):
        return '[60-70)'
    elif((x['age']=='[70-80)')):
        return '[70-80)'
    elif((x['age']=='[80-90)') | (x['age']=='[90-100)')):
        return '[80-100)'


# In[15]:


df['age']=df.apply(age_bins,axis=1)


# # Weight

# #### Around 97% of values are missing, it is better to drop this feature

# # Admission type id
# 
# Currently there are 8 sub-categories in the admission type_id
# 
# 1. Emergency	
# 2. Urgent	
# 3. Elective	
# 4. Newborn	
# 5. Not Available	
# 6. NULL	
# 7. Trauma Center	
# 8. Not Mapped

# There are few categories, where records are very less.
# To simplify this, we can club the categories which are similar in nature as shown below
# Emergency and Trauma centre can be clubbed as one category.
# 
# Urgent and Elective will be seperate categories.
# 
# New born and Not Mapped will be classified as others.
# 
# NA, Unknown will be in one category
# 
# New sub-categories are as follows:
# 1. Emergency
# 2. Urgency
# 3. Elective
# 4. Others
# 5. Unknown

# In[16]:


def ATI(x):
    if((x['admission_type_id']==1) | (x['admission_type_id']==7)):
        return 1
    elif((x['admission_type_id']==2)):
        return 2
    elif((x['admission_type_id']==3)):
        return 3
    elif((x['admission_type_id']==4) | (x['admission_type_id']==8)):
        return 4
    elif((x['admission_type_id']==5) | (x['admission_type_id']==6)):
        return 5
def ATI(x):
    if((x['admission_type_id']==1) | (x['admission_type_id']==7)):
        return 1
    elif((x['admission_type_id']==2)):
        return 2
    elif((x['admission_type_id']==3)):
        return 3
    elif((x['admission_type_id']==4) | (x['admission_type_id']==8)):
        return 4
    elif((x['admission_type_id']==5) | (x['admission_type_id']==6)):
        return 5
df['admission_type_id']=df.apply(ATI,axis=1)


# In[17]:


df['admission_type_id'].replace({1:'Emergency',2:'Urgency',3:'Elective',4:'Others',5:'Unknown'},inplace=True)


# # Discharge_disposition_ID

# There are very few categories, where records are very less, we can combine few categories here
# 
# There are 30 different categories in this particular column, let us try to bring down that to 7 columns.
# 
# ![title](discharge.png)
# ![title](capture1.png)

# In[18]:


def DDC(x):
    if(x['discharge_disposition_id']==1):
        return 1
    elif(x['discharge_disposition_id']==7):
        return 3
    elif((x['discharge_disposition_id']==9) | (x['discharge_disposition_id']==12) | (x['discharge_disposition_id']==15)):
        return 4
    elif((x['discharge_disposition_id']==11) | (x['discharge_disposition_id']==19) | (x['discharge_disposition_id']==20) | (x['discharge_disposition_id']==21)):
        return 5
    elif((x['discharge_disposition_id']==13) | (x['discharge_disposition_id']==14)):
        return 6
    elif((x['discharge_disposition_id']==18) | (x['discharge_disposition_id']==25) | (x['discharge_disposition_id']==26)):
        return 7
    else:
        return 2


# In[19]:


df['discharge_disposition_id']=df.apply(DDC,axis=1)


# In[20]:


df['discharge_disposition_id'].replace({1:'Discharged',2:'Transferred',3:'Left AMA',4:'Inpatient',5:'Expired',6:'Hospice',7:'Unknown'},inplace=True)


# # Admission Source ID

# There are 26 categories in this particular column, let us try to reduce the number of categories
# ![title](admsr1.png)
# ![title](admsr.png)

# In[21]:


def ASI(x):
    if(x['admission_source_id']==7):
        return 3
    elif((x['admission_source_id']==18) | (x['admission_source_id']==19)):
        return 6
    elif((x['admission_source_id']==1) | (x['admission_source_id']==2) | (x['admission_source_id']==3)):
        return 1
    elif((x['admission_source_id']==4) | (x['admission_source_id']==5) | (x['admission_source_id']==6) | (x['admission_source_id']==10) | (x['admission_source_id']==22) | (x['admission_source_id']==25) | (x['admission_source_id']==26)):
        return 2
    elif((x['admission_source_id']==9) | (x['admission_source_id']==15) | (x['admission_source_id']==17) | (x['admission_source_id']==20) | (x['admission_source_id']==21)):
        return 5
    else:
        return 4


# In[22]:


df['admission_source_id']=df.apply(ASI,axis=1)


# In[23]:


df['admission_source_id'].replace({1:'Referral',2:'Transfer',3:'Emergency',4:'Others',5:'Unknown',6:'Readmission'},inplace=True)


# # Diagnosis columns
# 
# #### Values which covered less than 3.5% where clubbed as 'other' categories
# 
# 1. Circulatory - 390–459, 785
# 2. Respiratory - 460–519, 786
# 3. Digestive - 520–579, 787
# 4. Diabetes - 250.xx
# 5. Injury 800–999
# 6. Musculoskeletal 710–739
# 7. Genitourinary 580–629, 788
# 8. Neoplasms 140–239
# 9. Other - 780, 781, 784, 790–799, 240–279, without 250, 680–709, 782, 001–139, 290–319, E–V, 280–289, 320–359, 630–679
# 360–389, 740–759

# In[24]:


# Treating the values in the diag_1, diag_2 and diag_3 columns
df["diag_1"]=list(map(lambda x: x.replace('V','10'),df["diag_1"]))
df["diag_1"]=list(map(lambda x: x.replace('E0','2000'),df["diag_1"]))
df["diag_1"]=list(map(lambda x: x.replace('E','200'),df["diag_1"]))
df["diag_2"]=list(map(lambda x: x.replace('E0','2000'),df["diag_2"]))
df["diag_2"]=list(map(lambda x: x.replace('E','20'),df["diag_2"]))
df["diag_2"]=list(map(lambda x: x.replace('V','10'),df["diag_2"]))
df["diag_3"]=list(map(lambda x: x.replace('E','2000'),df["diag_3"]))
df["diag_3"]=list(map(lambda x: x.replace('E','20'),df["diag_3"]))
df["diag_3"]=list(map(lambda x: x.replace('V','10'),df["diag_3"]))
# Replacing mising values with '?' with 0 and it needs no imputation as well
df["diag_1"]=list(map(lambda x: x.replace('?','0'),df["diag_1"]))
df["diag_2"]=list(map(lambda x: x.replace('?','0'),df["diag_2"]))
df["diag_3"]=list(map(lambda x: x.replace('?','0'),df["diag_3"]))


# In[25]:


# Converting the datatype of diag columns to float
df["diag_1"]=df["diag_1"].astype(float)
df["diag_2"]=df["diag_2"].astype(float)
df["diag_3"]=df["diag_3"].astype(float)


# In[26]:


# Converting them to floor values
df["diag_1"]=list(map(lambda x: np.floor(x),df["diag_1"]))
df["diag_2"]=list(map(lambda x: np.floor(x),df["diag_2"]))
df["diag_3"]=list(map(lambda x: np.floor(x),df["diag_3"]))


# In[27]:


# Converting them to int datatype
df["diag_1"]=df["diag_1"].astype(int)
df["diag_2"]=df["diag_2"].astype(int)
df["diag_3"]=df["diag_3"].astype(int)


# In[28]:


def diagnosis(a):
    if((a in range(390,460)) | (a==785)):
        return 'Circulatory'
    elif((a in range(460,520)) | (a==786)):
        return 'Respiratory'
    elif((a in range(520,580)) | (a==787)):
        return 'Digestive'
    elif (a == 250):
        return 'Diabetes'
    elif (a in range(800,1000)):
        return 'Injury'
    elif (a in range(710,740)):
        return 'Musculoskeletal'
    elif ((a in range(580,630)) | (a==788)):
        return 'Genitourinary'
    elif ((a in range(140,240))):
        return 'Neoplasms'
    elif ((a in range(290,320)) | (a in range(280,290)) | (a in range(320,360)) | (a in range(630,680)) | (a in range(360,390)) | (a in range(740,760)) | (a in range(1000,2000)) | (a >= 2000) | (a>2000) | (a in range(780,785)) | (a in range(789,800)) | (a in range (240,280)) | (a in range(680,710)) | (a in range(1,140))):
        return 'Others'
    elif ( a == 0):
        return 'Not Applicable'


# In[29]:


diag_1_list=list(map(diagnosis,df['diag_1']))
df['diag_1']=diag_1_list
diag_2_list=list(map(diagnosis,df['diag_2']))
df['diag_2']=diag_2_list
diag_3_list=list(map(diagnosis,df['diag_3']))
df['diag_3']=diag_3_list


# #### Since around 97% of values are missing in 'weight' attribute, we will drop it because it is too sparsed
# #### There are too many missing values in 'payer_code' attribute as well, hence we will drop those column from the analysis

# In[30]:


df.drop(['weight','payer_code'],axis=1,inplace=True)


# In[31]:


num_cols=['time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency',
          'number_inpatient','number_diagnoses']
cor_mat=df.corr()


# In[32]:


df=df.drop(['encounter_id', 'patient_nbr','medical_specialty'],axis=1)


# In[33]:


df['readmitted'].value_counts()


# Since there are very less number of records for early re-admission, it will be better to go for over sampling. We will consider 75:25 ratio for no readmissions and early early readmissions
# 
# We can drop encounter_id and patient_id from our data since they are just used for identification.
# We can drop weight, payer_code and medical_specialty because of large number of missing values.

# In[34]:


df_no=df[df['readmitted']=='NO']
df_30=df[df['readmitted']=='<30']


# In[35]:


l=30000
df_os=df_30.sample(l,replace=True,random_state=1234)
df=pd.concat([df_no,df_os],axis=0)


# In[36]:


df['readmitted'].value_counts()


# In[ ]:





# It can be seen that number of record sfor early readmission is increased to 30000 from 11357. We will be considering this data for further analysis.

# # Feature selection using Chi Square Test for categorical features

# In[37]:


cat_cols=list(df.columns[17:])


# In[38]:


cat_cols.extend(('race','gender', 'age','admission_type_id','discharge_disposition_id','admission_source_id','diag_1','diag_2',
                 'diag_3'))


# In[39]:


cat_cols.remove('readmitted')


# In[40]:


cat_cols.remove('Up_medicine')


# In[41]:


cat_cols.remove('Down_medicine')


# In[42]:


cat_cols.remove('Steady_medicine')


# In[43]:


cat_cols


# In[44]:


chi_stat=[]
p_value=[]
for i in cat_cols:
    chi_res=st.chi2_contingency(np.array(pd.crosstab(df[i],df['readmitted'])))
    chi_stat.append(chi_res[0])
    p_value.append(chi_res[1])


# In[45]:


chi_square=pd.DataFrame([chi_stat,p_value])
chi_square=chi_square.T
col=['Chi Square Value','P-Value']
chi_square.columns=col
chi_square.index=cat_cols


# In[46]:


#H0: Two attributes are independent
#H1: Two attributes are dependent.


# In[47]:


chi_square


# When two features are independent, the observed count is close to the expected count, thus we will have smaller Chi-Square value. So high Chi-Square value indicates that the hypothesis of independence is incorrect. In other words, higher the Chi-Square value the feature is more dependent on the response and it can be selected for model training. Higher chi square value implies lower P-Value
# 
# At 95% level of confidence

# In[48]:


chi_sel=chi_square[chi_square['P-Value']<=0.05]


# # ANOVA test for numerical features selection
# 
# ### If variance is low, it doesnt have impact on features.

# In[49]:


num_cols=['time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency',
         'number_inpatient','number_diagnoses','Up_medicine','Down_medicine','Steady_medicine']


# ## Assumptions for ANOVA test
# ### Assumption 1: Test for Normality (Shapiro)
# #### H0 : Population is normally distributed
# #### H1: Population is not Normally distributed
# 
# ### Assumption 2: Test for Variance (Levene)
# #### H0 : Population variance is equal
# #### H1: Population variance is not equal

# In[50]:


sp_val=[]
lp_val=[]
sf_stat=[]
lf_stat=[]
for i in num_cols:
    n1=df[df['readmitted']=='NO'][i]
    n2=df[df['readmitted']=='<30'][i]
    sh=st.shapiro(df[i])
    le=st.levene(n1,n2)
    sp_val.append(sh[1])
    sf_stat.append(sh[0])
    lp_val.append(le[1])
    lf_stat.append(le[0])
anova_assumptions=pd.DataFrame([sp_val,sf_stat,lp_val,lf_stat])
anova_assumptions=anova_assumptions.T
anova_assumptions.columns=['Shapiro P-Value','Shapiro F-Stat','Levene P-Value','Levene F-Stat']
anova_assumptions.index=num_cols
anova_assumptions


# #### In all of above features, P-Value of both Shapiro and Levene test is zero, hence we will reject Null hypothesis.
# #### None of the features pass  normality and variance test, hence we go for non-parametric test. Non parametric test is MannWhitneyU test

# In[51]:


m_val=[]
m_stat=[]
for i in num_cols:
    n1=df[df['readmitted']=='NO'][i]
    n2=df[df['readmitted']=='<30'][i]
    a=st.mannwhitneyu(n1,n2)
    m_val.append(a[1])
    m_stat.append(a[0])
mwu=(pd.DataFrame([m_val,m_stat])).T
mwu.columns=['P-Value','F_Stat']
mwu.index=num_cols


# In[52]:


mwu


# In[53]:


mwu.to_csv('mannwhitneyU.csv')


# In[54]:


f_stat=[]
p_val=[]
for i in num_cols:
    no_disc=df[df['readmitted']=='NO'][i]
    less_than_30=df[df['readmitted']=='<30'][i]
    a=st.f_oneway(no_disc,less_than_30)
    f_stat.append(a[0])
    p_val.append(a[1])


# In[55]:


anova=pd.DataFrame([f_stat,p_val])
anova=anova.T
cols=['F-STAT','P-VALUE']
anova.columns=cols
anova.index=num_cols


# #### Null hypothesis is rejected in all cases.
# #### All the above features have variance between them. Hence, these features can be used for model building

# #### Dropping Less significant features from model building

# In[56]:


dropped = ['gender','nateglinide','chlorpropamide','acetohexamide','tolbutamide','acarbose','troglitazone',
           'tolazamide','examide','citoglipton','glipizide-metformin','glimepiride-pioglitazone',
           'metformin-rosiglitazone','metformin-pioglitazone']
d=df.drop(dropped,axis=1)


# In[57]:


diab=d.copy()


# In[58]:


diab.head()


# # Treating Skewness of Numerical Variables

# In[59]:


for i in num_cols:
    a=diab[i].skew()
    print('Skewness for {} is {}\n'.format(i,a))


# #### We will cap the values of all numerical features at 95th percentile, any value which is above the value of 95th percentile will be made equal to 95th percentile.

# In[60]:


diab1=diab.copy()


# In[61]:


for i in num_cols:
    a=diab1[i].quantile(q=0.05)
    b=diab1[i].quantile(q=0.95)
    print('5th percentile for {} is {}'.format(i,a))
    print('95th percentile for {} is {}'.format(i,b))
    print('--------------------------------')


# In[62]:


def capping(x):
    if(x['time_in_hospital']>11):
        return 11
    else:
        return x['time_in_hospital']
diab1['time_in_hospital']=diab1.apply(capping,axis=1)
def capping1(x):
    if(x['num_lab_procedures']>73):
        return 73
    else:
        return x['num_lab_procedures']
diab1['num_lab_procedures']=diab1.apply(capping1,axis=1)
def capping2(x):
    if(x['num_procedures']>5):
        return 5
    else:
        return x['num_procedures']
diab1['num_procedures']=diab1.apply(capping2,axis=1)
def capping3(x):
    if(x['num_medications']>31):
        return 31
    else:
        return x['num_medications']
diab1['num_medications']=diab1.apply(capping3,axis=1)
def capping4(x):
    if(x['number_outpatient']>2):
        return 2
    else:
        return x['number_outpatient']
diab1['number_outpatient']=diab1.apply(capping4,axis=1)
def capping5(x):
    if(x['number_emergency']>1):
        return 1
    else:
        return x['number_emergency']
diab1['number_emergency']=diab1.apply(capping5,axis=1)
def capping6(x):
    if(x['number_inpatient']>3):
        return 3
    else:
        return x['number_inpatient']
diab1['number_inpatient']=diab1.apply(capping6,axis=1)
def capping7(x):
    if(x['number_diagnoses']>3):
        return 3
    else:
        return x['number_diagnoses']
diab1['number_diagnoses']=diab1.apply(capping7,axis=1)


# In[63]:


for i in num_cols:
    a=diab1[i].skew()
    print('Skewness for {} is {}\n'.format(i,a))


# ### Here we can see that, there is increase in skewness of some of the variables. Let us look at skewness after boxcox transformation

# In[64]:


l1=list((boxcox(diab1.time_in_hospital + 1)[0]))
l2=list((boxcox(diab1.num_lab_procedures + 1)[0]))
l3=list((boxcox(diab1.num_procedures + 1)[0]))
l4=list((boxcox(diab1.num_medications + 1)[0]))
l5=list((boxcox(diab1.number_outpatient + 1)[0]))
l6=list((boxcox(diab1.number_emergency + 1)[0]))
l7=list((boxcox(diab1.number_inpatient + 1)[0]))
l8=list((boxcox(diab1.number_diagnoses + 1)[0]))


# In[65]:


diab1['time_in_hospital']=l1
diab1['num_lab_procedures']=l2
diab1['num_procedures']=l3
diab1['num_medications']=l4
diab1['number_outpatient']=l5
diab1['number_emergency']=l6
diab1['number_inpatient']=l7
diab1['number_diagnoses']=l8


# In[66]:


for i in num_cols:
    a=diab1[i].skew()
    print('Skewness for {} is {}\n'.format(i,a))


# ### We can see that boxcox transformation reduces skewness in a better way compared to capping also no extreme records are not lost. Hence we will go with boxcox transformation

# In[67]:


l1=list((boxcox(diab.time_in_hospital + 1)[0]))
l2=list((boxcox(diab.num_lab_procedures + 1)[0]))
l3=list((boxcox(diab.num_procedures + 1)[0]))
l4=list((boxcox(diab.num_medications + 1)[0]))
l5=list((boxcox(diab.number_outpatient + 1)[0]))
l6=list((boxcox(diab.number_emergency + 1)[0]))
l7=list((boxcox(diab.number_inpatient + 1)[0]))
l8=list((boxcox(diab.number_diagnoses + 1)[0]))


# In[68]:


diab['time_in_hospital']=l1
diab['num_lab_procedures']=l2
diab['num_procedures']=l3
diab['num_medications']=l4
diab['number_outpatient']=l5
diab['number_emergency']=l6
diab['number_inpatient']=l7
diab['number_diagnoses']=l8


# In[69]:


for i in num_cols:
    a=diab[i].skew()
    print('Skewness for {} is {}\n'.format(i,a))


# # Encoding Categorical Data
# 
# ## Using binary encoding for age discharge deposition code,admission source id and admission type id
# ## For other features we will use one hot encoding.

# # Model building
# #### We will try fitting model using cross_val_score rather than train_test_split which will help in considering all the patterns in the data.

# In[70]:


def model_eval(algo,X_train,y_train,X_test,y_test):
    algo.fit(X_train,y_train)
    y_train_pred=algo.predict(X_train)
    y_train_prob=algo.predict_proba(X_train)[:,1]


    from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,roc_auc_score,roc_curve

    print('Overall Accuracy - train ',accuracy_score(y_train,y_train_pred))
    
    print('Confusion Matrix - train ','\n',confusion_matrix(y_train,y_train_pred))
    
    print('AUC -Train', roc_auc_score(y_train, y_train_prob))
    
    print('Classification Report for Train data:\n',classification_report(y_train,y_train_pred))
    #print(y_test_prob.shape, y_train_prob.shape)

    y_test_pred=algo.predict(X_test)
    y_test_prob=algo.predict_proba(X_test)[:,1]


    print('Overall Accuracy - test ',accuracy_score(y_test,y_test_pred))
    
    print('Confusion Matrix - test ','\n',confusion_matrix(y_test,y_test_pred))
    
    print('AUC -test', roc_auc_score(y_test, y_test_prob))
    
    print('Classification Report for Test data:\n',classification_report(y_test,y_test_pred))

    fpr,tpr,thresholds=roc_curve(y_test, y_test_prob)
    roc=pd.DataFrame({'fpr':fpr,'tpr':tpr,'thresholds':thresholds})
    fig,ax=plt.subplots()
    ax.plot(fpr,tpr)
    plt.plot(fpr,fpr,'r-')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    ax1=ax.twinx()
    ax1.plot(fpr,thresholds)
    ax1.set_ylabel('Thresholds')
    plt.show()


# # Label Encoder and Min-Max Scaling

# In[71]:


d1=d.copy()


# In[72]:


enc_col=['race', 'age', 'admission_type_id', 'discharge_disposition_id','admission_source_id','diag_1', 'diag_2', 'diag_3',
        'max_glu_serum', 'A1Cresult', 'metformin','repaglinide', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
       'rosiglitazone', 'miglitol', 'insulin', 'glyburide-metformin','change','diabetesMed']
le=LabelEncoder()
for i in enc_col:
    d1[i]=le.fit_transform(d1[i])


# In[73]:


d1['readmitted'].replace({'NO':0,'<30':1},inplace=True)


# In[74]:


x1=d1.drop('readmitted',axis=1)
y1=d1['readmitted']


# In[75]:


x_train1,x_test1,y_train1,y_test1=train_test_split(x1,y1,test_size=0.30,random_state=1234)


# In[76]:


min_max=MinMaxScaler()
x1s=min_max.fit_transform(x1)
x_train1s=min_max.fit_transform(x_train1)
x_test1s=min_max.transform(x_test1)


# In[77]:


x1s.shape,x1.shape


# In[78]:


x1s=pd.DataFrame(x1s,columns=x1.columns)


# In[79]:


rfc_best=RandomForestClassifier(n_estimators=180,min_samples_split=2,max_features=10,max_depth=14)


# In[80]:


model_eval(rfc_best,x_train1s,y_train1,x_test1s,y_test1)


# In[81]:


pickle.dump(rfc_best,open('deployment.pkl','wb'))

