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

#-------- TABLAS DE CONVENCIONES PARA EL DASH -------------#

# Datos para la tabla edad
datoss = [
    ['1-35', '1'],
    ['36-40','2'],
    ['41-45', '3'],
    ['46-50', '4'],
    ['51-55', '5'],
    ['56-60', '6'],
    ['61-65', '7'],
    ['66-70', '8'],
    ['71-75', '9'],
    ['Mayores de 75', '10'],
]
# Crear la tabla edad
tabla = html.Table([
    # Encabezados de las columnas
    html.Tr([html.Th('Age: Rango de Edad',style ={'text-align': 'center'}), html.Th('#',style ={'text-align': 'center'})], style = {'text-align': 'center','font-family': 'Lucida Bright, Georgia, serif', 'background-color': '#003085',
			'color': '#ffffff',
			'text-align': 'left',
			'font-weight': 'bold',
            'font-size': '1.0em',}),
    # Datos de las filas
    *[html.Tr([html.Td(d[0]), html.Td(d[1])], style ={'text-align': 'center'}) for d in datoss]
], style={'text-align': 'center','font-family': 'Lucida Bright, Georgia, serif', 'border-collapse': 'collapse',
			'margin': '30px',
			'font-size': '0.8em',
			'min-width': '180px',
			'overflow': 'hidden',
			'margin-top': '20px',
            'border':'1px solid #003085'})
# Datos para tabla Trestbps
datoss1 = [
    ['0-110', '1'],
    ['110-130','2'],
    ['130-150', '3'],
    ['150-170', '4'],
    ['Mayor a 170', '5']
]
# Crear la tabla Trestbps
tabla1 = html.Table([
    # Encabezados de las columnas
    html.Tr([html.Th('Trestbps: Rango de Trestbps',style ={'text-align': 'center'}), html.Th('#',style ={'text-align': 'center'})], style = {'text-align': 'center','font-family': 'Lucida Bright, Georgia, serif', 'background-color': '#003085',
			'color': '#ffffff',
			'text-align': 'left',
			'font-weight': 'bold',
            'font-size': '1.0em'}),
    # Datos de las filas
    *[html.Tr([html.Td(d[0]), html.Td(d[1])], style ={'text-align': 'center'}) for d in datoss1]
], style={'text-align': 'center','font-family': 'Lucida Bright, Georgia, serif', 'border-collapse': 'collapse',
			'margin': '30px',
			'font-size': '0.8em',
			'min-width': '180px',
			'overflow': 'hidden',
			'margin-top': '20px',
            'border':'1px solid #003085'})
# Datos colesterol
datoss2 = [
    ['0-140', '1'],
    ['140-180','2'],
    ['180-220', '3'],
    ['220-260', '4'],
    ['260-300', '5'],
    ['300-340', '6'],
    ['340-380', '7'],
    ['380-420', '8'],
    ['420-460', '9'],
    ['Mayor a 460', '10']
]
# Crear la tabla Colesterol
tabla2 = html.Table([
    # Encabezados de las columnas
    html.Tr([html.Th('Chol: Rango de Colesterol',style ={'text-align': 'center'}), html.Th('#',style ={'text-align': 'center'})], style = {'text-align': 'center','font-family': 'Lucida Bright, Georgia, serif', 'background-color': '#003085',
			'color': '#ffffff',
			'text-align': 'left',
			'font-weight': 'bold',
            'font-size': '1.0em',}),
    # Datos de las filas
    *[html.Tr([html.Td(d[0]), html.Td(d[1])], style ={'text-align': 'center'}) for d in datoss2]
], style={'text-align': 'center','font-family': 'Lucida Bright, Georgia, serif', 'border-collapse': 'collapse',
			'margin': '30px',
			'font-size': '0.8em',
			'min-width': '180px',
			'overflow': 'hidden',
			'margin-top': '20px',
            'border':'1px solid #003085'})
# Datos fbs
datoss3 = [
    ['Mayor a 120 mg/dl', '1'],
    ['Menor o igual a 120 mg/dl','0']
]
# Crear la tabla fbs
tabla3 = html.Table([
    # Encabezados de las columnas
    html.Tr([html.Th('fbs: Muestra de Azúcar',style ={'text-align': 'center'}), html.Th('#',style ={'text-align': 'center'})], style = {'text-align': 'center','font-family': 'Lucida Bright, Georgia, serif', 'background-color': '#003085',
			'color': '#ffffff',
			'text-align': 'left',
			'font-weight': 'bold',
            'font-size': '1.0em',}),
    # Datos de las filas
    *[html.Tr([html.Td(d[0]), html.Td(d[1])], style ={'text-align': 'center'}) for d in datoss3]
], style={'text-align': 'center','font-family': 'Lucida Bright, Georgia, serif', 'border-collapse': 'collapse',
			'margin': '30px',
			'font-size': '0.8em',
			'min-width': '180px',
			'overflow': 'hidden',
			'margin-top': '20px',
            'border':'1px solid #003085'})
# Datos sex
datoss4 = [
    ['Hombre', '1'],
    ['Mujer','0']
]
# Crear la tabla sex
tabla4 = html.Table([
    # Encabezados de las columnas
    html.Tr([html.Th('Sex: Sexo del paciente',style ={'text-align': 'center'}), html.Th('#',style ={'text-align': 'center'})], style = {'text-align': 'center','font-family': 'Lucida Bright, Georgia, serif', 'background-color': '#003085',
			'color': '#ffffff',
			'text-align': 'left',
			'font-weight': 'bold',
            'font-size': '1.0em',}),
    # Datos de las filas
    *[html.Tr([html.Td(d[0]), html.Td(d[1])], style ={'text-align': 'center'}) for d in datoss4]
], style={'text-align': 'center','font-family': 'Lucida Bright, Georgia, serif', 'border-collapse': 'collapse',
			'margin': '30px',
			'font-size': '0.8em',
			'min-width': '180px',
			'overflow': 'hidden',
			'margin-top': '20px',
            'border':'1px solid #003085'})

# Datos exang
datoss5 = [
    ['Experimentó angina', '1'],
    ['No experimentó angina','0']
]
# Crear la tabla exang
tabla5 = html.Table([
    # Encabezados de las columnas
    html.Tr([html.Th('Exang: Angina',style ={'text-align': 'center'}), html.Th('#',style ={'text-align': 'center'})], style = {'text-align': 'center','font-family': 'Lucida Bright, Georgia, serif', 'background-color': '#003085',
			'color': '#ffffff',
			'text-align': 'left',
			'font-weight': 'bold',
            'font-size': '1.0em',}),
    # Datos de las filas
    *[html.Tr([html.Td(d[0]), html.Td(d[1])], style ={'text-align': 'center'}) for d in datoss5]
], style={'text-align': 'center','font-family': 'Lucida Bright, Georgia, serif', 'border-collapse': 'collapse',
			'margin': '30px',
			'font-size': '0.8em',
			'min-width': '180px',
			'overflow': 'hidden',
			'margin-top': '20px',
            'border':'1px solid #003085'})

# Datos cp
datoss6 = [
    ['Angina típica', '1'],
    ['Angina atípica','2'],
    ['Dolor no anginal','3'],
    ['Asintomático','4']
]
# Crear la tabla cp
tabla6 = html.Table([
    # Encabezados de las columnas
    html.Tr([html.Th('CP: Dolor de pecho',style ={'text-align': 'center'}), html.Th('#',style ={'text-align': 'center'})], style = {'text-align': 'center','font-family': 'Lucida Bright, Georgia, serif', 'background-color': '#003085',
			'color': '#ffffff',
			'text-align': 'left',
			'font-weight': 'bold',
            'font-size': '1.0em',}),
    # Datos de las filas
    *[html.Tr([html.Td(d[0]), html.Td(d[1])], style ={'text-align': 'center'}) for d in datoss6]
], style={'text-align': 'center','font-family': 'Lucida Bright, Georgia, serif', 'border-collapse': 'collapse',
			'margin': '30px',
			'font-size': '0.8em',
			'min-width': '180px',
			'overflow': 'hidden',
			'margin-top': '20px',
            'border':'1px solid #003085'})

# Datos restecg
datoss7 = [
    ['ECG normal', '0'],
    ['Anomalía onda ST-T','1'],
    ['Hipertrofia venticular','2']
]
# Crear la tabla restecg
tabla7 = html.Table([
    # Encabezados de las columnas
    html.Tr([html.Th('Restecg: ECG en reposo',style ={'text-align': 'center'}), html.Th('#',style ={'text-align': 'center'})], style = {'text-align': 'center','font-family': 'Lucida Bright, Georgia, serif', 'background-color': '#003085',
			'color': '#ffffff',
			'text-align': 'left',
			'font-weight': 'bold',
            'font-size': '1.0em',}),
    # Datos de las filas
    *[html.Tr([html.Td(d[0]), html.Td(d[1])], style ={'text-align': 'center'}) for d in datoss7]
], style={'text-align': 'center','font-family': 'Lucida Bright, Georgia, serif', 'border-collapse': 'collapse',
			'margin': '30px',
			'font-size': '0.8em',
			'min-width': '180px',
			'overflow': 'hidden',
			'margin-top': '20px',
            'border':'1px solid #003085'})


# Datos thalach
datoss8 = [
    ['1-100', '1'],
    ['101-110','2'],
    ['111-120','3'],
    ['121-130', '4'],
    ['131-140','5'],
    ['141-150','6'],
    ['151-160', '7'],
    ['161-170','8'],
    ['171-180','9'],
    ['181-190', '10'],
    ['Mayores a 190','11']    
]
# Crear la tabla thalach
tabla8 = html.Table([
    # Encabezados de las columnas
    html.Tr([html.Th('Thalach: Frecuencia max alcanzada',style ={'text-align': 'center'}), html.Th('#',style ={'text-align': 'center'})], style = {'text-align': 'center','font-family': 'Lucida Bright, Georgia, serif', 'background-color': '#003085',
			'color': '#ffffff',
			'text-align': 'left',
			'font-weight': 'bold',
            'font-size': '1.0em',}),
    # Datos de las filas
    *[html.Tr([html.Td(d[0]), html.Td(d[1])], style ={'text-align': 'center'}) for d in datoss8]
], style={'text-align': 'center','font-family': 'Lucida Bright, Georgia, serif', 'border-collapse': 'collapse',
			'margin': '30px',
			'font-size': '0.8em',
			'min-width': '180px',
			'overflow': 'hidden',
			'margin-top': '20px',
            'border':'1px solid #003085'})



# Ruta de la imagen:
cora = 'https://images.emojiterra.com/google/android-11/512px/1fac0.png'
uniandes = 'https://uniandes.edu.co/sites/default/files/logo-uniandes.png'



#-----------------------FRONT DE LA APLICACIÓN---------------------------#


app.layout = html.Div([
    html.Div([ html.Div([html.Img(src=cora, height='120px', width='120px'),], style={'display': 'inline-block', 'vertical-align': 'middle', 'margin-right': '20px'}),
    html.Div([html.H1("DETECCIÓN DE PROBABILIDAD DE PADECER ENFERMEDAD DEL CORAZÓN (BN)", style={'font-family': 'Lucida Bright, Georgia, serif', 'text-align': 'center','color':'#000000','margin-left':'70px'})], style={'display': 'inline-block', 'vertical-align': 'middle'}),
    html.Div([html.Img(src=uniandes, height='150px', width='240px'),], style={'display': 'inline-block', 'vertical-align': 'middle', 'margin-left': '20px'}),
    ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'text-align': 'center'}),    
    html.Hr(style={'border': '3px solid black', 'margin-top':'-5px'}),
    html.H2("Ingresar valores para la consulta:", style={'font-family': 'Lucida Bright, Georgia, serif', 'text-align': 'left', 'color':'#FF4720', 'margin-left':'100px'}),
    html.Div([
   dcc.Input(
    id="age",
    type="number",
    min=1,
    max=10,
    placeholder="Age",
    style={
       'font-family': 'Lucida Bright, Georgia, serif',
        'border': '2px solid',
        'padding': '6px 8px',
        'text-align': 'center',
        'font-size': '1.1em',
        'margin-top': '20px',
        }
),
    dcc.Input(
    id="trestbps",
    type="number",
    min=1,
    max=5,
    placeholder="Trestbps",
    style={
        'font-family': 'Lucida Bright, Georgia, serif',
        'border': '2px solid',
        'padding': '6px 8px',
        'text-align': 'center',
        'font-size': '1.1em',
    }
),
    dcc.Input(
    id="chol",
    type="number",
    min=0,
    max=10,
    placeholder="Chol",
    style={
        'font-family': 'Lucida Bright, Georgia, serif',
        'border': '2px solid',
        'padding': '6px 8px',
        'text-align': 'center',
        'font-size': '1.1em',
    }
),
    dcc.Input(
    id="fbs",
    type="number",
    min=0,
    max=1,
    placeholder="Fbs",
    style={
        'font-family': 'Lucida Bright, Georgia, serif',
        'border': '2px solid',
        'padding': '6px 8px',
        'text-align': 'center',
        'font-size': '1.1em',
    }
),
    dcc.Input(
    id="sex",
    type="number",
    min=0,
    max=1,
    placeholder="Sex",
    style={
        'font-family': 'Lucida Bright, Georgia, serif',
        'border': '2px solid',
        'padding': '6px 8px',
        'text-align': 'center',
        'font-size': '1.1em',
    }
), html.Br(),
    dcc.Input(
    id="restecg",
    type="number",
    min=0,
    max=2,
    placeholder="Restecg",
    style={
        'font-family': 'Lucida Bright, Georgia, serif',
        'border': '2px solid',
        'padding': '6px 8px',
        'text-align': 'center',
        'font-size': '1.1em',
    }
),
    dcc.Input(
    id="thalach",
    type="number",
    min=1,
    max=11,
    placeholder="Thalach",
    style={
        'font-family': 'Lucida Bright, Georgia, serif',
        'border': '2px solid',
        'padding': '6px 8px',
        'text-align': 'center',
        'font-size': '1.1em',
    }
),
  dcc.Input(
    id="cp",
    type="number",
    min=1,
    max=4,
    placeholder="CP",
    style={
        'font-family': 'Lucida Bright, Georgia, serif',
        'border': '2px solid',
        'padding': '6px 8px',
        'text-align': 'center',
        'font-size': '1.1em',
    }
),
  dcc.Input(
    id="exang",
    type="number",
    min=0,
    max=1,
    placeholder="Exang",
    style={
        'font-family': 'Lucida Bright, Georgia, serif',
        'border': '2px solid',
        'padding': '6px 8px',
        'text-align': 'center',
        'font-size': '1.1em',
    }
), ],style={'margin-left':'220px', 'margin-top':'-10px'}),
    
    html.Div([html.Br(),
        html.Button(
        "Consultar",
        id="btn",
        style={
            'background-color': '#a4a5a4',
            'border': 'none',
            'color': '#fff',
            'cursor': 'pointer',
            'font-size': '1.3em',
            'margin': '10px 10px',
            'padding': '6px 8px',
            'transition': 'background-color 0.3s ease',
            'left': '2px',
            'top': '30px',
            'font-family': 'Lucida Bright, Georgia, serif',
        }
    )],style={'margin-left':'1085px', 'margin-top':'-58px'}),
    html.Div([    html.H3("Convenciones para la herramienta:", style={'font-family': 'Lucida Bright, Georgia, serif', 'text-align': 'left','color':'#003085','margin':'15px', 'margin-top':'15px', 'margin-left':'100px'}),    
    html.Div([tabla, tabla2, tabla1, tabla3, tabla4], style={'display': 'flex','align-items': 'center','margin-left':'120px','margin-top':'-15px'}),
    html.Div([tabla5, tabla6, tabla7, tabla8],style={'display': 'flex','align-items': 'center', 'margin-left':'250px','margin-top':'-40px'}),
    html.H1(id="output",style={'font-family': 'Lucida Bright, Georgia, serif','text-align': 'center','color':'#50B452','margin':'15px', 'margin-top':'20px'})
], style={'text-align': 'right'}),
])
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
    

