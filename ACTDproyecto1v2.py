#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD


# # IMPORTAR DATAFRAME Y VER DATOS

# In[3]:


df_Cleveland = pd.read_csv('C:\\Users\\juand\\OneDrive\\Documents\\Andes\\10. Semestre\\Analítica computacional\\Proyectos\\Proyecto 1\\processed.cleveland.data',
                           names =["Age","Sex","CP","Trestbps","Chol","Fbs","Restecg","Thalach","Exang","Oldpeak","Slope","Ca","Thal","Num"])


# In[ ]:


df_Cleveland


# In[ ]:


#df_Usar = df_Cleveland[["Age","Sex","Trestbps","Chol","Fbs","Restecg","Num"]]


# In[ ]:


#df_Usar.describe()


# # GRAFICAR LOS DATOS.

# In[34]:


plt.bar(df_Cleveland["Age"], height = df_Cleveland["Chol"])


# In[35]:


plt.bar(df_Cleveland["Age"], height = df_Cleveland["Trestbps"])


# In[36]:


plt.scatter(df_Cleveland["Age"],df_Cleveland["Trestbps"])


# In[37]:


plt.scatter(df_Cleveland["Age"],df_Cleveland["Chol"])


# # AGRUPAR DATOS
# 
# # CAMBIAR VALORES DE PRESIÓN ARTERIAL

# In[4]:


# para valores entre 0 y 110, 1
#para valores entre 110 y 130, 2
#para valores entre 130 y 150, 3
#para valores entre 150 y 170, 4
#para valores entre 170 y infiito, 5
df_Cleveland["Trestbps"] = np.where((df_Cleveland["Trestbps"]<=110)&(df_Cleveland["Trestbps"]>1),1,df_Cleveland["Trestbps"])
df_Cleveland["Trestbps"] = np.where((df_Cleveland["Trestbps"]<=130)&(df_Cleveland["Trestbps"]>110),2,df_Cleveland["Trestbps"])
df_Cleveland["Trestbps"] = np.where((df_Cleveland["Trestbps"]<=150)&(df_Cleveland["Trestbps"]>130),3,df_Cleveland["Trestbps"])
df_Cleveland["Trestbps"] = np.where((df_Cleveland["Trestbps"]<=170)&(df_Cleveland["Trestbps"]>150),4,df_Cleveland["Trestbps"])
df_Cleveland["Trestbps"] = np.where((df_Cleveland["Trestbps"]<=10000)&(df_Cleveland["Trestbps"]>170),5,df_Cleveland["Trestbps"])

df_Cleveland.describe()


# # CAMBIAR VALOR DE AÑOS

# In[5]:


df_Cleveland["Age"] = np.where((df_Cleveland["Age"]<=35)&(df_Cleveland["Age"]>1),1,df_Cleveland["Age"])
df_Cleveland["Age"] = np.where((df_Cleveland["Age"]<=40)&(df_Cleveland["Age"]>35),2,df_Cleveland["Age"])
df_Cleveland["Age"] = np.where((df_Cleveland["Age"]<=45)&(df_Cleveland["Age"]>40),3,df_Cleveland["Age"])
df_Cleveland["Age"] = np.where((df_Cleveland["Age"]<=50)&(df_Cleveland["Age"]>45),4,df_Cleveland["Age"])
df_Cleveland["Age"] = np.where((df_Cleveland["Age"]<=55)&(df_Cleveland["Age"]>50),5,df_Cleveland["Age"])
df_Cleveland["Age"] = np.where((df_Cleveland["Age"]<=60)&(df_Cleveland["Age"]>55),6,df_Cleveland["Age"])
df_Cleveland["Age"] = np.where((df_Cleveland["Age"]<=65)&(df_Cleveland["Age"]>60),7,df_Cleveland["Age"])
df_Cleveland["Age"] = np.where((df_Cleveland["Age"]<=70)&(df_Cleveland["Age"]>65),8,df_Cleveland["Age"])
df_Cleveland["Age"] = np.where((df_Cleveland["Age"]<=75)&(df_Cleveland["Age"]>70),9,df_Cleveland["Age"])
df_Cleveland["Age"] = np.where((df_Cleveland["Age"]<=8000)&(df_Cleveland["Age"]>75),10,df_Cleveland["Age"])



df_Cleveland.describe()


# # CAMBIAR VALOR DE COLESTEROL.

# In[6]:


# para valores entre 0 y 140, 1
#para valores entre 140 y 180, 2
#para valores entre 180 y 220, 3
#para valores entre 220 y 260, 4
#para valores entre 260 y 300, 5
#para valores entre 300 y 340, 6
#para valores entre 340 y 380, 7
#para valores entre 380 y 420, 8
#para valores entre 420 y 460, 9
#para valores entre 500 y infinito, 10


df_Cleveland["Chol"] = np.where((df_Cleveland["Chol"]<=140)&(df_Cleveland["Chol"]>1),1,df_Cleveland["Chol"])
df_Cleveland["Chol"] = np.where((df_Cleveland["Chol"]<=180)&(df_Cleveland["Chol"]>140),2,df_Cleveland["Chol"])
df_Cleveland["Chol"] = np.where((df_Cleveland["Chol"]<=220)&(df_Cleveland["Chol"]>180),3,df_Cleveland["Chol"])
df_Cleveland["Chol"] = np.where((df_Cleveland["Chol"]<=260)&(df_Cleveland["Chol"]>220),4,df_Cleveland["Chol"])
df_Cleveland["Chol"] = np.where((df_Cleveland["Chol"]<=300)&(df_Cleveland["Chol"]>260),5,df_Cleveland["Chol"])
df_Cleveland["Chol"] = np.where((df_Cleveland["Chol"]<=340)&(df_Cleveland["Chol"]>300),6,df_Cleveland["Chol"])
df_Cleveland["Chol"] = np.where((df_Cleveland["Chol"]<=380)&(df_Cleveland["Chol"]>340),7,df_Cleveland["Chol"])
df_Cleveland["Chol"] = np.where((df_Cleveland["Chol"]<=420)&(df_Cleveland["Chol"]>380),8,df_Cleveland["Chol"])
df_Cleveland["Chol"] = np.where((df_Cleveland["Chol"]<=460)&(df_Cleveland["Chol"]>420),9,df_Cleveland["Chol"])
df_Cleveland["Chol"] = np.where((df_Cleveland["Chol"]<=8000)&(df_Cleveland["Chol"]>460),10,df_Cleveland["Chol"])

df_Cleveland


# In[7]:


df_Cleveland.describe()


# In[8]:


df_usar = df_Cleveland[["Age","Sex","Trestbps","Chol","Fbs","Restecg","Num"]]


# In[9]:


df_usar.describe()


# # HACER RED BAYESIANA

# In[10]:


from pgmpy.sampling import BayesianModelSampling
samples = df_usar
print (samples.head())


# In[11]:


model = BayesianNetwork ([("Age", "Trestbps") , ("Age", "Chol"),("Sex", "Trestbps"),
                           ("Sex", "Chol"),("Age", "Fbs"),("Trestbps", "Num"),
                           ("Chol", "Num"),("Fbs","Num")])


# In[12]:


from pgmpy.estimators import MaximumLikelihoodEstimator
emv = MaximumLikelihoodEstimator( model,data=samples)


# In[13]:


cpdem_Age = emv.estimate_cpd(node="Age")
print (cpdem_Age)
cpdem_Sex = emv.estimate_cpd(node="Sex")
print (cpdem_Sex)

cpdem_Trestbps = emv.estimate_cpd(node="Trestbps")
print (cpdem_Trestbps)
cpdem_Chol = emv.estimate_cpd(node="Chol")
print (cpdem_Chol)
cpdem_Fbs = emv.estimate_cpd(node="Fbs")
print (cpdem_Fbs)
cpdem_Num = emv.estimate_cpd(node="Num")
print (cpdem_Num)


# In[14]:


model.add_cpds(cpdem_Age , cpdem_Sex , cpdem_Trestbps ,cpdem_Chol,cpdem_Fbs,cpdem_Num )


# In[16]:


model.check_model()


# In[17]:


from pgmpy.inference import VariableElimination
infer = VariableElimination (model)


# In[18]:


print ( model . get_independencies () )


# In[21]:


# ME LLAMAN JUAN Y MARIA
posterior_p1 = infer.query(["Num"] , evidence ={"Age": 4 , "Sex": 1,"Trestbps":2,"Chol":7,"Fbs":1},)
print ( posterior_p1 )


# In[32]:


posterior_p2 = infer.query(["Num"] , evidence ={"Age":8,"Trestbps": 5 , "Chol": 5,"Fbs":1,},)
print ( posterior_p2 )

