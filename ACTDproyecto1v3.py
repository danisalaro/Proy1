# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 19:39:49 2023

@author: JUAN y DAN
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

#############################----------------------------------------- SECCIÓN DE ANÁLISIS DESCRIPTIVO DE LOS DATOS Y GRÁFICAS--------------------------------------######

#-------------------HISTOGRAMAS-------------------------#

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

app.layout = html.Div([
    html.H1("HERRAMIENTA PARA LA DETECCIÓN DE PROBABILIDAD DE PADECER ENFERMEDAD DEL CORAZÓN", style={'font-family': 'Poppins, sans-serif', 'text-align': 'center','color':'#003085','margin':'15px'}),
    html.H2("Ingresar valores para la consulta", style={'font-family': 'Poppins, sans-serif', 'text-align': 'center', 'color':'#FF4720'}),
    dcc.Input(id="age", type="number", placeholder="Age", style={'font-family': 'Poppins, sans-serif', 'border': '2px solid', 'padding': '6px 8px', 'text-align': 'center','font-size': '1.1em','margin-top':'30px', 'margin-left': '120px'}),
    dcc.Input(id="trestbps", type="number", placeholder="Trestbps", style={'font-family': 'Poppins, sans-serif', 'border': '2px solid', 'padding': '6px 8px', 'text-align': 'center','font-size': '1.1em'}),
    dcc.Input(id="chol", type="number", placeholder="Chol", style={'font-family': 'Poppins, sans-serif', 'border': '2px solid', 'padding': '6px 8px', 'text-align': 'center','font-size': '1.1em'}),
    dcc.Input(id="fbs", type="number", placeholder="Fbs", style={'font-family': 'Poppins, sans-serif', 'border': '2px solid', 'padding': '6px 8px', 'text-align': 'center','font-size': '1.1em'}),
    dcc.Input(id="sex", type="number", placeholder="Sex", style={'font-family': 'Poppins, sans-serif', 'border': '2px solid', 'padding': '6px 8px', 'text-align': 'center','font-size': '1.1em'}),
    html.Button(
        "Consultar",
        id="btn",
        style={
        'background-color': '#a4a5a4',
        'border': 'none',
        'border-radius': '40px',
        'color': '#fff',
        'cursor': 'pointer',
        'font-size': '1.2em',
        'margin': '10px 0',
        'padding': '10px 20px',
        'transition': 'background-color 0.3s ease',
        'position': 'absolute'
    }
),

    html.Br(),
    html.P("Si la edad está entre 1 y 30 ingrese: 1.", style={'font-family': 'Poppins, sans-serif'}),
    html.P("Si la edad está entre 30 y 35 ingrese: 2." , style={'font-family': 'Poppins, sans-serif'}),
    html.P("Si la edad está entre 40 y 45 ingrese: 3.", style={'font-family': 'Poppins, sans-serif'}),
    html.P("Si la edad está entre 50 y 55 ingrese: 4." , style={'font-family': 'Poppins, sans-serif'}),
    html.P("Si la edad está entre 55 y 60 ingrese: 5." , style={'font-family': 'Poppins, sans-serif'}),
    html.P("Si la edad está entre 60 y 65 ingrese: 6." , style={'font-family': 'Poppins, sans-serif'}),
    html.P("Si la edad está entre 65 y 70 ingrese: 7.", style={'font-family': 'Poppins, sans-serif'}),
    html.P("Si la edad es mayor a 70 ingrese: 8", style={'font-family': 'Poppins, sans-serif'}),
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
    
    
    
    