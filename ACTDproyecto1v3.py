# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 19:39:49 2023

@author: JUAN DIEGO PRADA y DANIEL FELIPE SALAZAR
"""

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
    

df_Cleveland = pd.read_csv('cleveland.data',
                           names =["Age","Sex","CP","Trestbps","Chol","Fbs","Restecg","Thalach","Exang","Oldpeak","Slope","Ca","Thal","Num"])

print(df_Cleveland)
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

# Age con Chol

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
# Age con Oldpeak:

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

# Age con Thalach:

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


df_usar = df_Cleveland[["Age","Sex","Trestbps","Chol","Fbs","Restecg","Num"]]

from pgmpy.sampling import BayesianModelSampling
samples = df_usar
print (samples.head())

model = BayesianNetwork ([("Age", "Trestbps") , ("Age", "Chol"),("Sex", "Trestbps"),
                           ("Sex", "Chol"),("Age", "Fbs"),("Trestbps", "Num"),
                           ("Chol", "Num"),("Fbs","Num")])

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


model.add_cpds(cpdem_Age , cpdem_Sex , cpdem_Trestbps ,cpdem_Chol,cpdem_Fbs,cpdem_Num )

from pgmpy.inference import VariableElimination
infer = VariableElimination (model)






app = dash.Dash(__name__)

# Datos para la tabla
datoss = [
    ['1-30', '1'],
    ['31-39','2'],
    ['40-49', '3'],
    ['50-55', '4'],
    ['56-59', '5'],
    ['60-65', '6'],
    ['65-70', '7'],
    ['Mayor a 70', '8'],
]

# Crear la tabla
tabla = html.Table([
    # Encabezados de las columnas
    html.Tr([html.Th('Rango de Edad',style ={'text-align': 'center'}), html.Th('Número a Digitar',style ={'text-align': 'center'})], style = {'text-align': 'center','font-family': 'Poppins, sans-serif', 'background-color': '#003085',
			'color': '#ffffff',
			'text-align': 'left',
			'font-weight': 'bold'}),
    # Datos de las filas
    *[html.Tr([html.Td(d[0]), html.Td(d[1])], style ={'text-align': 'center'}) for d in datoss]
], style={'text-align': 'center','font-family': 'Poppins, sans-serif', 'border-collapse': 'collapse',
			'margin': '30px',
			'font-size': '1.2em',
			'min-width': '400px',
			'overflow': 'hidden',
			'margin-top': '20px',
            'border':'1px solid #003085'})

# Ruta de la imagen:
cora = 'https://www.shutterstock.com/image-vector/valentines-day-heart-vector-illustration-260nw-556690450.jpg'

app.layout = html.Div([
    html.Img(src= cora, height='40px', width='50px'),
    html.H1("HERRAMIENTA PARA LA DETECCIÓN DE PROBABILIDAD DE PADECER ENFERMEDAD DEL CORAZÓN", style={'font-family': 'Poppins, sans-serif', 'text-align': 'center','color':'#003085','margin':'15px'}),
    html.H2("Ingresar valores para la consulta:", style={'font-family': 'Poppins, sans-serif', 'text-align': 'center', 'color':'#FF4720'}),
    dcc.Input(id="age", type="number", min=1, max=8, placeholder="Age", style={'font-family': 'Poppins, sans-serif', 'border': '2px solid', 'padding': '6px 8px', 'text-align': 'center','font-size': '1.1em','margin-top':'30px', 'margin-left': '120px'}),
    dcc.Input(id="trestbps", type="number",min=1, max=5, placeholder="Trestbps", style={'font-family': 'Poppins, sans-serif', 'border': '2px solid', 'padding': '6px 8px', 'text-align': 'center','font-size': '1.1em'}),
    dcc.Input(id="chol", type="number",min=0,max=10, placeholder="Chol", style={'font-family': 'Poppins, sans-serif', 'border': '2px solid', 'padding': '6px 8px', 'text-align': 'center','font-size': '1.1em'}),
    dcc.Input(id="fbs", type="number",min=0, max=1, placeholder="Fbs", style={'font-family': 'Poppins, sans-serif', 'border': '2px solid', 'padding': '6px 8px', 'text-align': 'center','font-size': '1.1em'}),
    dcc.Input(id="sex", type="number",min=0, max=1,placeholder="Sex", style={'font-family': 'Poppins, sans-serif', 'border': '2px solid', 'padding': '6px 8px', 'text-align': 'center','font-size': '1.1em'}),
    html.Button(
        "Consultar",
        id="btn",
        style={
        'background-color': '#a4a5a4',
        'border': 'none',
        'color': '#fff',
        'cursor': 'pointer',
        'font-size': '1.2em',
        'margin': '10px 0',
        'padding': '6px 8px',
        'transition': 'background-color 0.3s ease',
        'position': 'absolute',
        'margin-left': '2px',
        'margin-top': '30px',
    }
),
    html.H3("Convenciones para la herramienta:",style={'font-family': 'Poppins, sans-serif', 'text-align': 'left','color':'#003085','margin':'15px', 'margin-top':'15px'}),
    tabla,
    html.Div(id="output"),
])


@app.callback(Output("output", "children"), Input("btn", "n_clicks"),
              Input("age", "value"), Input("trestbps", "value"),
              Input("chol", "value"), Input("fbs", "value"),
              Input("sex","value"))
def run_query(n_clicks, age, trestbps, chol, fbs,sex):
    if n_clicks is not None:
        posterior_p2 = infer.query(["Num"], evidence={"Age": age, "Trestbps": trestbps, "Chol": chol, "Fbs": fbs,"Sex": sex})
        return f"El resultado de la consulta es: {posterior_p2}"  


if __name__ == '__main__':
    app.run_server(debug=False)
    