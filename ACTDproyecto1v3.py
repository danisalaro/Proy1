# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 19:39:49 2023

@author: JUAN DIEGO PRADA y DANIEL FELIPE SALAZAR
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
    

df_Cleveland = pd.read_csv('cleveland.data',
                           names =["Age","Sex","CP","Trestbps","Chol","Fbs","Restecg","Thalach","Exang","Oldpeak","Slope","Ca","Thal","Num"])

print(df_Cleveland)
'''
#############################----------------------------------------- SECCIÓN DE ANÁLISIS DESCRIPTIVO DE LOS DATOS Y GRÁFICAS--------------------------------------######

#-------------------HISTOGRAMAS-------------------------#


# Thalach:

hist0 = px.histogram(df_Cleveland, x="Thalach", title="HISTOGRAMA DE MÁXIMO RATE REGISTRADO",
                labels={"Thalach": "Nivel Registrado", "count": "Frecuencia",})

hist0.update_layout(
    title={
        'text': "HISTOGRAMA DE MÁXIMO RATE REGISTRADO",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28, color='black')
    },
    xaxis_title="Nivel Registrado",
    yaxis_title="Frecuencia",
    font=dict(size=18, color='black'),
    plot_bgcolor='white',
    bargap=0.1,
    margin=dict(l=50, r=50, t=100, b=50),
    showlegend=False
)
hist0.update_traces(marker_color='navy')
hist0.show()

# Colesterol:    

hist = px.histogram(df_Cleveland, x="Chol", title="HISTOGRAMA DE COLESTEROL",
                labels={"Chol": "Niveles de Colesterol", "count": "Frecuencia",})

hist.update_layout(
    title={
        'text': "HISTOGRAMA DE COLESTEROL",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28, color='black')
    },
    xaxis_title="Niveles de Colesterol",
    yaxis_title="Frecuencia",
    font=dict(size=18, color='black'),
    plot_bgcolor='white',
    bargap=0.1,
    margin=dict(l=50, r=50, t=100, b=50),
    showlegend=False
)
hist.update_traces(marker_color='darkgreen')
hist.show()

# Edad:
hist1 = px.histogram(df_Cleveland, x="Age", title="HISTOGRAMA DE EDAD",
                labels={"Age": "Edad", "count": "Frecuencia"})

hist1.update_layout(
    title={
        'text': "Histograma de Edad",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28, color='black')
    },
    xaxis_title="Edad",
    yaxis_title="Frecuencia",
    font=dict(size=18, color='black'),
    plot_bgcolor='white',
    bargap=0.1,
    margin=dict(l=50, r=50, t=100, b=50),
    showlegend=False
)

hist1.update_traces(marker_color='#7F3C8D')

hist1.show()

# Oldpeak:
hist2 = px.histogram(df_Cleveland, x="Oldpeak", title="HISTOGRAMA DE ST DEPRESSION",
                labels={"Oldpeak": "St Depression", "count": "Frecuencia"})

hist2.update_layout(
    title={
        'text': "Histograma de ST Depression",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28, color='black')
    },
    xaxis_title="ST Depression",
    yaxis_title="Frecuencia",
    font=dict(size=18, color='black'),
    plot_bgcolor='white',
    bargap=0.1,
    margin=dict(l=50, r=50, t=100, b=50),
    showlegend=False
)

hist2.update_traces(marker_color='#FF966F')

hist2.show()
# Ca:    
hist3 = px.histogram(df_Cleveland, x="Ca", title="HISTOGRAMA DE VASOS COLOREADOS",
                labels={"Ca": "Número de vasos sanguineos coloreados", "count": "Frecuencia"})

hist3.update_layout(
    title={
        'text': "HISTOGRAMA DE VASOS COLOREADOS",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28, color='black')
    },
    xaxis_title="Vasos coloreados",
    yaxis_title="Frecuencia",
    font=dict(size=18, color='black'),
    plot_bgcolor='white',
    bargap=0.1,
    margin=dict(l=50, r=50, t=100, b=50),
    showlegend=False
)

hist3.update_traces(marker_color='#8B98B2')

hist3.show()

# Trestbps:    
hist4 = px.histogram(df_Cleveland, x="Trestbps", title="HISTOGRAMA DE PRESIÓN SANGUINEA EN REPOSO",
                labels={"Trestbps": "Presión sanguinea en reposo", "count": "Frecuencia"})

hist4.update_layout(
    title={
        'text': "HISTOGRAMA DE PRESIÓN SANGUINEA EN REPOSO",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28, color='black')
    },
    xaxis_title="Presión sanguinea",
    yaxis_title="Frecuencia",
    font=dict(size=18, color='black'),
    plot_bgcolor='white',
    bargap=0.1,
    margin=dict(l=50, r=50, t=100, b=50),
    showlegend=False
)

hist4.update_traces(marker_color='#599B86')


#-------------------PIE CHARTS-------------------------#
# Hombres y mujeres
datos_pie = {'Sexo': ['Hombres', 'Mujeres'], 'Cantidad': [df_Cleveland['Sex'].value_counts()[1], df_Cleveland['Sex'].value_counts()[0]]}
pie_chart1 = px.pie(datos_pie, values='Cantidad', names='Sexo', 
             color_discrete_sequence=['#2D7BB6','#F7BFBE'],  # Cambiar los colores de las secciones de la torta
             hole=0.5,  # Agregar un agujero en el centro de la torta
             title='DISTRIBUCIÓN DE GÉNERO',  # Agregar un título al gráfico
             labels={'Cantidad': 'Cantidad de personas', 'Sexo': 'Género'},  # Cambiar los nombres de los ejes
             template='seaborn',  # Cambiar el estilo del gráfico
             )

pie_chart1.update_layout(
    legend=dict(
        x=0.5,
        y=0.9,
        traceorder='normal',
        font=dict(
            size=14,
        ),
    ),
)
pie_chart1.show()

# Azúcar en sangre tomada en ayunas:

datos_pie1 = {'Azucar': ['Menor a 120mg/dl ', 'Mayor a 120mg/dl '], 'Cantidad': [df_Cleveland['Fbs'].value_counts()[0], df_Cleveland['Fbs'].value_counts()[1]]}
pie_chart2 = px.pie(datos_pie1, values='Cantidad', names='Azucar', 
             color_discrete_sequence=['#2D7BB6','#FF0000'],  # Cambiar los colores de las secciones de la torta
             hole=0.5,  # Agregar un agujero en el centro de la torta
             title='DISTRIBUCIÓN DE TOMAS DE AZÚCAR EN SANGRE',  # Agregar un título al gráfico
             labels={'Cantidad': 'Cantidad de personas', 'Azucar': 'Cantidad de azucar'},  # Cambiar los nombres de los ejes
             template='seaborn',  # Cambiar el estilo del gráfico
             )

pie_chart2.update_layout(
    legend=dict(
        x=0.5,
        y=0.9,
        traceorder='normal',
        font=dict(
            size=14,
        ),
    ),
)
pie_chart2.show()

print(df_Cleveland)

#-------------------SCATTER Y RELACIÓN ENTRE VARIABLES-------------------------#

# Age con Chol (sexo)

sc0=px.scatter(df_Cleveland, y='Chol', x= 'Age', title = 'RELACIÓN ENTRE COLESTEROL Y EDAD',labels={"Chol": "Colesterol registrado", "Age": "Edad"},color="Sex", facet_col="Sex")
sc0.update_layout(
    title={
        'text': "RELACIÓN ENTRE COLESTEROL Y EDAD",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=17, color='black')
    }
)
sc0.show()
# Age con Oldpeak (sexo):

sc1=px.scatter(df_Cleveland, y='Oldpeak', x= 'Age', title = 'RELACIÓN ENTRE ST DEPRESSION Y EDAD',labels={"Oldpeak": "ST Depression", "Age": "Edad"},color="Sex", facet_col="Sex")
sc1.update_layout(
    title={
        'text': "RELACIÓN ENTRE ST DEPRESSION Y EDAD",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=17, color='black')
    }
)
sc1.show()

# Age con Thalach (enfermedad):

sc2=px.scatter(df_Cleveland, y='Thalach', x= 'Age', title = 'RELACIÓN ENTRE MÁXIMO RATE REGISTRADO Y EDAD',labels={"Thalach": "Máximo rate registrado", "Age": "Edad"},color="Sex", facet_col="Sex")
sc2.update_layout(
    title={
        'text': "RELACIÓN ENTRE MÁXIMO RATE REGISTRADO Y EDAD",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=17, color='black')
    }
)
sc2.show()

# Colesterol con Thalach (enfermedad)
sc3=px.scatter(df_Cleveland, y='Chol', x= 'Thalach', title = 'RELACIÓN ENTRE MÁXIMO RATE REGISTRADO Y COLESTEROL',labels={"Chol": "Colesterol", "Thalach": "Máximo rate registrado"},color="Num", facet_col="Num")
sc3.update_layout(
    title={
        'text': "RELACIÓN ENTRE MÁXIMO RATE REGISTRADO Y COLESTEROL",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=17, color='black')
    }
)
sc3.show()

# Age con Chol (enfermedad)

sc4=px.scatter(df_Cleveland, y='Chol', x= 'Age', title = 'RELACIÓN ENTRE COLESTEROL Y EDAD',labels={"Chol": "Colesterol registrado", "Age": "Edad"},color="Num", facet_col="Num")
sc4.update_layout(
    title={
        'text': "RELACIÓN ENTRE COLESTEROL Y EDAD",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=17, color='black')
    }
)
sc4.show()

# Age con Oldpeak (enfermedad):

sc5=px.scatter(df_Cleveland, y='Oldpeak', x= 'Age', title = 'RELACIÓN ENTRE ST DEPRESSION Y EDAD',labels={"Oldpeak": "ST Depression", "Age": "Edad"},color="Num", facet_col="Num")
sc5.update_layout(
    title={
        'text': "RELACIÓN ENTRE ST DEPRESSION Y EDAD",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=17, color='black')
    }
)
sc5.show()

# Age con Trestbps (sexo)
sc6=px.scatter(df_Cleveland, y='Trestbps', x= 'Age', title = 'RELACIÓN ENTRE PRESIÓN EN SANGRE Y EDAD',labels={"Trestbps": "Presión en sangre", "Age": "Edad"},color="Sex", facet_col="Sex")
sc6.update_layout(
    title={
        'text': "RELACIÓN ENTRE PRESIÓN EN SANGRE Y EDAD",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=17, color='black')
    }
)
sc6.show()

# Age con Trestbps (enfermedad)
sc7=px.scatter(df_Cleveland, y='Trestbps', x= 'Age', title = 'RELACIÓN ENTRE PRESIÓN EN SANGRE Y EDAD',labels={"Trestbps": "Presión en sangre", "Age": "Edad"},color="Num", facet_col="Num")
sc7.update_layout(
    title={
        'text': "RELACIÓN ENTRE PRESIÓN EN SANGRE Y EDAD",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=17, color='black')
    }
)
sc7.show()


#------------ MAPA DE CALOR -----------------#

matrizcorr = df_Cleveland.corr()

mpc = px.imshow(matrizcorr, 
                 x=matrizcorr.columns, 
                 y=matrizcorr.columns,
                 color_continuous_scale='RdBu')
mpc.update_layout(title='Matriz de Correlación para el Dataset Cleveland',
                   xaxis_title='Variables',
                   yaxis_title='Variables')
mpc.show()
'''

#############################------------------------------------------BAYESIAN NETWORK----------------------------------------------##########


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
#Cambiar valores para año
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
# CAMBIAR VALORES COLESTEROL
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


df_Cleveland["Thalach"] = np.where((df_Cleveland["Thalach"]<=100)&(df_Cleveland["Thalach"]>1),1,df_Cleveland["Thalach"])
df_Cleveland["Thalach"] = np.where((df_Cleveland["Thalach"]<=110)&(df_Cleveland["Thalach"]>100),2,df_Cleveland["Thalach"])
df_Cleveland["Thalach"] = np.where((df_Cleveland["Thalach"]<=120)&(df_Cleveland["Thalach"]>110),3,df_Cleveland["Thalach"])
df_Cleveland["Thalach"] = np.where((df_Cleveland["Thalach"]<=130)&(df_Cleveland["Thalach"]>120),4,df_Cleveland["Thalach"])
df_Cleveland["Thalach"] = np.where((df_Cleveland["Thalach"]<=140)&(df_Cleveland["Thalach"]>130),5,df_Cleveland["Thalach"])
df_Cleveland["Thalach"] = np.where((df_Cleveland["Thalach"]<=150)&(df_Cleveland["Thalach"]>140),6,df_Cleveland["Thalach"])
df_Cleveland["Thalach"] = np.where((df_Cleveland["Thalach"]<=160)&(df_Cleveland["Thalach"]>150),7,df_Cleveland["Thalach"])
df_Cleveland["Thalach"] = np.where((df_Cleveland["Thalach"]<=170)&(df_Cleveland["Thalach"]>160),8,df_Cleveland["Thalach"])
df_Cleveland["Thalach"] = np.where((df_Cleveland["Thalach"]<=180)&(df_Cleveland["Thalach"]>170),9,df_Cleveland["Thalach"])
df_Cleveland["Thalach"] = np.where((df_Cleveland["Thalach"]<=190)&(df_Cleveland["Thalach"]>180),10,df_Cleveland["Thalach"])
df_Cleveland["Thalach"] = np.where((df_Cleveland["Thalach"]<=2000)&(df_Cleveland["Thalach"]>190),11,df_Cleveland["Thalach"])







df_usar = df_Cleveland[["Age","Sex","Trestbps","Chol","Fbs","Restecg","Num","Thalach","CP","Exang"]]
from pgmpy.sampling import BayesianModelSampling
samples = df_usar
print (samples.head())



model = BayesianNetwork ([("Age", "Chol") , ("Age", "Fbs"),("Age", "Restecg"),("Age", "Thalach"),
                           ("Sex", "Chol"),("Sex", "Fbs"), ("Sex", "Restecg"),("Sex", "Thalach"),
                           ("Chol","Num"),
                           ("Fbs","Num"),
                           ("Restecg","Num"),
                           ("Thalach", "Num"),
                           ("Num","Trestbps"), ("Num","CP"),("Num","Exang")])



from pgmpy.estimators import MaximumLikelihoodEstimator
emv = MaximumLikelihoodEstimator( model,data=samples)
cpdem_Age = emv.estimate_cpd(node="Age")
#print (cpdem_Age)
cpdem_Sex = emv.estimate_cpd(node="Sex")
#print (cpdem_Sex)
cpdem_Trestbps = emv.estimate_cpd(node="Trestbps")
#print (cpdem_Trestbps)
cpdem_Chol = emv.estimate_cpd(node="Chol")
#print (cpdem_Chol)
cpdem_Fbs = emv.estimate_cpd(node="Fbs")
#print (cpdem_Fbs)
cpdem_Num = emv.estimate_cpd(node="Num")
#print (cpdem_Num)
cpdem_Restecg = emv.estimate_cpd(node="Restecg")
#print (cpdem_Restecg)
cpdem_Thalach = emv.estimate_cpd(node="Thalach")
#print (cpdem_Thalach)
cpdem_CP = emv.estimate_cpd(node="CP")
#print (cpdem_CP)
cpdem_Exang = emv.estimate_cpd(node="Exang")
#print (cpdem_Exang)


model.add_cpds(cpdem_Age , cpdem_Sex , cpdem_Trestbps ,cpdem_Chol,cpdem_Fbs,cpdem_Num,cpdem_Restecg,cpdem_Thalach,cpdem_CP,cpdem_Exang )
from pgmpy.inference import VariableElimination
infer = VariableElimination (model)
app = dash.Dash(__name__)

# Ruta de la imagen:
cora = 'https://images.emojiterra.com/google/android-11/512px/1fac0.png'
uniandes = 'https://uniandes.edu.co/sites/default/files/logo-uniandes.png'



#-----------------------FRONT DE LA APLICACIÓN---------------------------#
import dash_bootstrap_components as dbc
app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

app.css.append_css({
    'external_url': 'https://bootswatch.com/4/darkly/bootstrap.css'
})

bienvenida = dbc.Card(
    dbc.CardBody(
        [
            html.Div([
                html.H1("PROYECTO ANALÍTICA COMPUTACIONAL PARA LA TOMA DE DECISIONES", className="text-center"),
                html.Br(),
                html.Br(),
                html.P("El objetivo del presente proyecto consiste en atender las necesidades de los médicos especialistas con el fin de poder, a través de redes bayesianas identificar si el paciente padece una enefermedad del corazón. Para este se utilizó la base de datos de la Universidad de California en Irvine, en donde se presenta el resultado de un estudio realizado en Cleveland, Ohio. Los datos pueden ser escargados a través del siguiente enlace: https://archive-beta.ics.uci.edu/dataset/45/heart+disease", style={'text-align':'justify'}),
                html.Div([
                        html.Img(src=uniandes)
                    ], style={'text-align': 'center'}),
                html.Br(),
                html.P("Proyecto realizado por: Juan Diego Prada y Daniel Felipe Salazar"),
                html.H4("DISCLAIMER: El presente proyecto no representa a la Universidad de los Andes ni sus intereses.", style={'color': 'red', 'text-align':'center'}, className="fst-italic")
        ], style={'margin':'30px'})
            
        ]
    ),
    className="mt-3", 
    style={
        'border': 'none', 
        'background-color': 'transparent',
        'width': '100%', 
        'height': '100%', 
        'display': 'flex', 
        'justify-content': 'center', 
        'align-items': 'center',
        'flex-direction': 'column'
    }
)

tab1_content = dbc.Card(
    dbc.CardBody(
        [
                html.Br(),
                dbc.Row([
                    dbc.Row(
                    
                        dbc.Col(html.H1("HERRAMIENTA PARA LA DETECCIÓN DE ENFERMEDADES DEL CORAZÓN", className="text-center", style={'margin-top':'-20px'}))),
                    html.Br(),
                    html.Br(),
                    dbc.Col(html.Div([ 
                        html.P("Por favor seleccione los datos del paciente:"),
                        dcc.Dropdown(
                        id="age",
                        options=[
                                {'label': '1-35', 'value': 1},
                                {'label': '36-40', 'value': 2},
                                {'label': '41-45', 'value': 3},
                                {'label': '46-50', 'value': 4},
                                {'label': '51-55', 'value': 5},
                                {'label': '56-60', 'value': 6},
                                {'label': '61-65', 'value': 7},
                                {'label': '66-70', 'value': 8},
                                {'label': '71-75', 'value': 9},
                                {'label': 'Mayores de 75', 'value': 10},
                                ],
                        placeholder = "Edad"
                    ),
                        dcc.Dropdown(
                        id="trestbps",
                        options=[
                                {'label': '0-110', 'value': 1},
                                {'label': '110-130', 'value': 2},
                                {'label': '130-150', 'value': 3},
                                {'label': '150-170', 'value': 4},
                                {'label': 'Mayor a 170', 'value': 5},
                                ],
                        placeholder="Trestbps",
                    
                    ),
                        dcc.Dropdown(
                        id="chol",
                        options=[
                                {'label': '0-140', 'value': 1},
                                {'label': '140-180', 'value': 2},
                                {'label': '180-220', 'value': 3},
                                {'label': '220-260', 'value': 4},
                                {'label': '260-300', 'value': 5},
                                {'label': '300-340', 'value': 6},
                                {'label': '340-380', 'value': 7},
                                {'label': '380-420', 'value': 8},
                                {'label': '420-460', 'value': 9},
                                {'label': 'Mayor a 460', 'value': 10},
                                ],
                        placeholder="Chol",
                        
                    ),
                        dcc.Dropdown(
                        id="fbs",
                        options=[
                                {'label': 'Mayor a 120 mg/dl', 'value': 1},
                                {'label': 'Menor o igual a 120 mg/dl', 'value': 0},
                                ],
                        placeholder="Fbs",
                        
                    ),
                        dcc.Dropdown(
                        id="sex",
                            options=[
                                {'label': 'Hombre', 'value': 1},
                                {'label': 'Mujer', 'value': 0},
                                ],
                        placeholder="Sex",
                    
                    ),
                        dcc.Dropdown(
                        id="restecg",
                        options=[
                                {'label': 'ECG normal', 'value': 0},
                                {'label': 'Anomalía onda ST-T', 'value': 1},
                                {'label': 'Hipertrofia venticular', 'value': 2},
                                ],
                        placeholder="Restecg",
                    
                    ),
                        dcc.Dropdown(
                        id="thalach",
                        options=[
                                {'label': '1-100', 'value': 1},
                                {'label': '101-110', 'value': 2},
                                {'label': '111-120', 'value': 3},
                                {'label': '121-130', 'value': 4},
                                {'label': '131-140', 'value': 5},
                                {'label': '141-150', 'value': 6},
                                {'label': '151-160', 'value': 7},
                                {'label': '161-170', 'value': 8},
                                {'label': '171-180', 'value': 9},
                                {'label': '181-190', 'value': 10},
                                {'label': 'Mayores a 190', 'value': 11},
                                ],
                        placeholder="Thalach",

                    ),
                    dcc.Dropdown(
                        id="cp",
                        options=[
                                {'label': 'Angina típica', 'value': 1},
                                {'label': 'Angina atípica', 'value': 2},
                                {'label': 'Dolor no anginal', 'value': 3},
                                {'label': 'Asintomático', 'value': 4},
                                ],
                        placeholder="CP",
                        
                    ),
                    dcc.Dropdown(
                        id="exang",
                        options=[
                                {'label': 'Experimentó angina', 'value': 1},
                                {'label': 'No experimentó angina', 'value': 0},
                                ],
                        placeholder="Exang",
                        
                    )]), width=6),
                    
                ]),
                html.Div([
                    html.Br(),
                    dbc.Button("Consultar", color="dark", className="me-1", id='btn')
                ]),
                html.Div([
                    html.Br(),
                    html.H4(id="output")
                , ])
        ]
    ),
    className="mt-3", style={'border': 'none', 'background-color': 'transparent','width': '100%', 'height': '100%'}
)

tab2_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 2!", className="card-text"),
            dbc.Button("Don't click here", color="danger"),
        ]
    ),
    className="mt-3", style={'border': 'none', 'background-color': 'transparent','width': '100%', 'height': '100%'}
)


tabs = dbc.Tabs(
    [
        dbc.Tab(bienvenida, label="Bienvenida"),
        dbc.Tab(tab1_content, label="Interfaz"),
        dbc.Tab(tab2_content, label="Gráficos de Interés"),
        
    ]
)

app.layout = html.Div([
    tabs,    
],style={"margin-top": "30px", "margin-left":"60px"})

@app.callback(Output("output", "children"), Input("btn", "n_clicks"),
              Input("age", "value"), Input("trestbps", "value"),
              Input("chol", "value"), Input("fbs", "value"),
              Input("sex","value"),Input("restecg","value"),Input("thalach","value"),
              Input("cp","value"),Input("exang","value"))
def run_query(n_clicks, age, trestbps, chol, fbs, sex, restecg, thalach,cp,exang):
    if n_clicks is not None:
        posterior_p2 = infer.query(["Num"], evidence={"Age": age, "Trestbps": trestbps, "Chol": chol, "Fbs": fbs, "Sex": sex,"Restecg": restecg,"Thalach": thalach,"CP": cp,"Exang": exang})
        suma = posterior_p2.values[1]+posterior_p2.values[2]+posterior_p2.values[3]+posterior_p2.values[4]
        return f"La probabilidad del paciente de tener la enfermedad es de: {round(suma*100,2)}%"
if __name__ == '__main__':
    app.run_server(debug=False)
    

