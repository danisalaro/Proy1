# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:50:08 2023

@author: JUAN
"""

"""
PROYECTO REALIZIADO POR: JUAN DIEGO PRADA y DANIEL FELIPE SALAZAR
"""

import dash
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from dash import dcc
import plotly.express as px
import plotly.graph_objects as go
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.readwrite import XMLBIFWriter
    

df_Cleveland = pd.read_csv('cleveland.data',
                           names =["Age","Sex","CP","Trestbps","Chol","Fbs","Restecg","Thalach","Exang","Oldpeak","Slope","Ca","Thal","Num"])

print(df_Cleveland)

################################################################################################
#--------------------------------------VISUALIZACIONES-----------------------------------------#
###############################################################################################

# Crear las visualizaciones

# Visualización 1: MAPA DE CALOR

matrizcorr = df_Cleveland.corr()

mpc = px.imshow(matrizcorr, 
                    x=matrizcorr.columns, 
                    y=matrizcorr.columns,
                    color_continuous_scale='RdBu')

mpc.update_layout(
        title='MAPA DE CALOR DE LOS DATOS',
        xaxis_title='Variables',
        yaxis_title='Variables',
        font_color='black'
    )
mpc.update_layout(
    width=1000,
    height=500
)


# Crear una nueva columna "Target" que represente si tiene o no enfermedad
df_Cleveland['Target'] = df_Cleveland['Num'].map(lambda x: 1 if x>0 else 0)


# Promedio de Age por enfermedad y sex
df_mean_age = df_Cleveland.groupby(["Target", "Sex"], as_index=False)["Age"].mean()

# Iterar sobre los dos valores posibles de sex para crear cada gráfico

fig = px.bar(df_mean_age, x="Target", y="Age", title="PROMEDIO DE EDAD SEGÚN PADECIMIENTO ENFERMEDAD",range_y=(0,75), 
             color=df_mean_age['Sex'].apply(lambda x: 'Hombre' if x>0 else 'Mujer'), 
             barmode='group')
    
fig.update_layout(
    xaxis_title="",
    yaxis_title="Promedio de Edad",
    font=dict(size=12, color='black'),
    legend=dict(title='', font=dict(size=12)),
    margin=dict(t=80),
    plot_bgcolor='white',
    coloraxis=dict(colorbar=dict(title='', tickfont=dict(size=12)), colorbar_len=0.3),
)
fig.update_xaxes(range=[-0.5, 1.5], tickvals=[0, 1], ticktext=['No tiene la enfermedad', 'Tiene la enfermedad'])
fig.update_layout(
    width=1000,
    height=500
)

# Maximo rate registrado promedio con enfermedad para hombres y mujeres

# Promedio de Age por enfermedad y sex
df_mean_th = df_Cleveland.groupby(["Target", "Age"], as_index=False)["Thalach"].mean()

fig1 = px.line(df_mean_th, x="Age", y="Thalach", title="PROMEDIO DE MÁXIMO RITMO CARDIACO ALCANZADO SEGÚN PADECIMIENTO ENFERMEDAD",
             color=df_mean_th['Target'].apply(lambda x: 'Tiene la enfermedad' if x>0 else 'No tiene la enfermedad'))
    
fig1.update_layout(
    xaxis_title="Edad",
    yaxis_title="Ritmo Cardiaco Promedio Alcanzado",
    font=dict(size=12, color='black'),
    legend=dict(title='', font=dict(size=12)),
    margin=dict(t=80),
    plot_bgcolor='white',
    coloraxis=dict(colorbar=dict(title='', tickfont=dict(size=12)), colorbar_len=0.3)
)
fig1.update_layout(
    width=1000,
    height=500
)

#################################################################################################################
#--------------------------------------CONSTRUCCIÓN DE LA RED BAYESIANA-----------------------------------------#
#################################################################################################################


from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:proyecto2@proyecto2.cflupen1v64d.us-east-1.rds.amazonaws.com:5432/proyecto2')

samples = pd.read_sql('SELECT * FROM proy2', con=engine)

from pgmpy.sampling import BayesianModelSampling
   

from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score
scoring_method = K2Score(data=samples)
esth = HillClimbSearch(data=samples)
estimated_modelh = esth.estimate(
    scoring_method=scoring_method, max_indegree=7, max_iter=int(1e4),fixed_edges = {("Age","Chol"),("Sex","Chol"),("Chol","Num")},
                                                                                    black_list = {("Trestbps","Age")})
#
print(estimated_modelh)
print(estimated_modelh.nodes())
print(estimated_modelh.edges())
import networkx as nx
import matplotlib.pyplot as plt

# Crear el gráfico dirigido del modelo
G = nx.DiGraph()
for parent, child in estimated_modelh.edges():
    G.add_edge(parent, child)

# Dibujar el gráfico del modelo
pos = nx.spring_layout(G, seed=42) # Asignar posiciones a los nodos
nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500) # Dibujar nodos
nx.draw_networkx_edges(G, pos, arrows=True) # Dibujar arcos con flechas
nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif") # Agregar etiquetas de nodos

plt.axis("off") # Ocultar los ejes
plt.title("Modelo Estimado")
plt.show() # Mostrar el gráfico


from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

estimated_model = BayesianNetwork(estimated_modelh)
estimated_model.fit(data=samples, estimator = MaximumLikelihoodEstimator) 
for i in estimated_model.nodes():
    print(estimated_model.get_cpds(i))


    
from pgmpy.inference import VariableElimination
infer = VariableElimination (estimated_model)

# write model to an XML BIF file 
writer = XMLBIFWriter(estimated_model)
writer.write_xmlbif('monty.xml')
   


