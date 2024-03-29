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

# Visualización 2: RELACIÓN ENFERMEDAD CON EDAD Y COLESTEROL

sc4 = px.scatter(df_Cleveland, x='Chol', y='Age', title='RELACIÓN ENTRE COLESTEROL Y EDAD', 
                 labels={"Chol": "Colesterol registrado", "Age": "Edad"}, 
                 color=df_Cleveland['Sex'].apply(lambda x: 'Hombre' if x>0 else 'Mujer'),
                 facet_col=df_Cleveland["Sex"].apply(lambda x: 'Hombre' if x>0 else 'Mujer'),
                 )

sc4.update_layout(
    title={
        'text': "RELACIÓN ENTRE COLESTEROL Y EDAD POR SEXO",
        'font': dict(size=17, color='black')
    }
)
# Crear una nueva columna "Target" que represente si tiene o no enfermedad
df_Cleveland['Target'] = df_Cleveland['Num'].map(lambda x: 1 if x>0 else 0)


# Promedio de Age por enfermedad y sex
df_mean_age = df_Cleveland.groupby(["Target", "Sex"], as_index=False)["Age"].mean()

# Iterar sobre los dos valores posibles de sex para crear cada gráfico

fig = px.bar(df_mean_age, x="Target", y="Age", title="PROMEDIO DE EDAD SEGÚN PADECIMIENTO ENFERMEDAD", facet_col="Sex", 
             color=df_mean_age['Sex'].apply(lambda x: 'Hombre' if x>0 else 'Mujer'), 
             barmode='group')
    
fig.update_layout(
    xaxis_title="Tiene la enfermedad",
    yaxis_title="Promedio de Edad",
    font=dict(size=12, color='black'),
    legend=dict(title='', font=dict(size=12)),
    margin=dict(t=80),
    plot_bgcolor='white',
    coloraxis=dict(colorbar=dict(title='', tickfont=dict(size=12)), colorbar_len=0.3)
)

fig.update_xaxes(range=[-0.5, 1.5], tickvals=[0, 1], ticktext=['No tiene la enfermedad', 'Tiene la enfermedad'])

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


#############################------------------------------------------BAYESIAN NETWORK----------------------------------------------##########
df_mod = df_Cleveland

# para valores entre 0 y 110, 1
#para valores entre 110 y 130, 2
#para valores entre 130 y 150, 3
#para valores entre 150 y 170, 4
#para valores entre 170 y infiito, 5

df_mod["Trestbps"] = np.where((df_mod["Trestbps"]<=110)&(df_mod["Trestbps"]>1),1,df_mod["Trestbps"])
df_mod["Trestbps"] = np.where((df_mod["Trestbps"]<=130)&(df_mod["Trestbps"]>110),2,df_mod["Trestbps"])
df_mod["Trestbps"] = np.where((df_mod["Trestbps"]<=150)&(df_mod["Trestbps"]>130),3,df_mod["Trestbps"])
df_mod["Trestbps"] = np.where((df_mod["Trestbps"]<=170)&(df_mod["Trestbps"]>150),4,df_mod["Trestbps"])
df_mod["Trestbps"] = np.where((df_mod["Trestbps"]<=10000)&(df_mod["Trestbps"]>170),5,df_mod["Trestbps"])

#valores para edad

df_mod["Age"] = np.where((df_mod["Age"]<=35)&(df_mod["Age"]>1),1,df_mod["Age"])
df_mod["Age"] = np.where((df_mod["Age"]<=40)&(df_mod["Age"]>35),2,df_mod["Age"])
df_mod["Age"] = np.where((df_mod["Age"]<=45)&(df_mod["Age"]>40),3,df_mod["Age"])
df_mod["Age"] = np.where((df_mod["Age"]<=50)&(df_mod["Age"]>45),4,df_mod["Age"])
df_mod["Age"] = np.where((df_mod["Age"]<=55)&(df_mod["Age"]>50),5,df_mod["Age"])
df_mod["Age"] = np.where((df_mod["Age"]<=60)&(df_mod["Age"]>55),6,df_mod["Age"])
df_mod["Age"] = np.where((df_mod["Age"]<=65)&(df_mod["Age"]>60),7,df_mod["Age"])
df_mod["Age"] = np.where((df_mod["Age"]<=70)&(df_mod["Age"]>65),8,df_mod["Age"])
df_mod["Age"] = np.where((df_mod["Age"]<=75)&(df_mod["Age"]>70),9,df_mod["Age"])
df_mod["Age"] = np.where((df_mod["Age"]<=8000)&(df_mod["Age"]>75),10,df_mod["Age"])

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

df_mod["Chol"] = np.where((df_mod["Chol"]<=140)&(df_mod["Chol"]>1),1,df_mod["Chol"])
df_mod["Chol"] = np.where((df_mod["Chol"]<=180)&(df_mod["Chol"]>140),2,df_mod["Chol"])
df_mod["Chol"] = np.where((df_mod["Chol"]<=220)&(df_mod["Chol"]>180),3,df_mod["Chol"])
df_mod["Chol"] = np.where((df_mod["Chol"]<=260)&(df_mod["Chol"]>220),4,df_mod["Chol"])
df_mod["Chol"] = np.where((df_mod["Chol"]<=300)&(df_mod["Chol"]>260),5,df_mod["Chol"])
df_mod["Chol"] = np.where((df_mod["Chol"]<=340)&(df_mod["Chol"]>300),6,df_mod["Chol"])
df_mod["Chol"] = np.where((df_mod["Chol"]<=380)&(df_mod["Chol"]>340),7,df_mod["Chol"])
df_mod["Chol"] = np.where((df_mod["Chol"]<=420)&(df_mod["Chol"]>380),8,df_mod["Chol"])
df_mod["Chol"] = np.where((df_mod["Chol"]<=460)&(df_mod["Chol"]>420),9,df_mod["Chol"])
df_mod["Chol"] = np.where((df_mod["Chol"]<=8000)&(df_mod["Chol"]>460),10,df_mod["Chol"])


df_mod["Thalach"] = np.where((df_mod["Thalach"]<=100)&(df_mod["Thalach"]>1),1,df_mod["Thalach"])
df_mod["Thalach"] = np.where((df_mod["Thalach"]<=110)&(df_mod["Thalach"]>100),2,df_mod["Thalach"])
df_mod["Thalach"] = np.where((df_mod["Thalach"]<=120)&(df_mod["Thalach"]>110),3,df_mod["Thalach"])
df_mod["Thalach"] = np.where((df_mod["Thalach"]<=130)&(df_mod["Thalach"]>120),4,df_mod["Thalach"])
df_mod["Thalach"] = np.where((df_mod["Thalach"]<=140)&(df_mod["Thalach"]>130),5,df_mod["Thalach"])
df_mod["Thalach"] = np.where((df_mod["Thalach"]<=150)&(df_mod["Thalach"]>140),6,df_mod["Thalach"])
df_mod["Thalach"] = np.where((df_mod["Thalach"]<=160)&(df_mod["Thalach"]>150),7,df_mod["Thalach"])
df_mod["Thalach"] = np.where((df_mod["Thalach"]<=170)&(df_mod["Thalach"]>160),8,df_mod["Thalach"])
df_mod["Thalach"] = np.where((df_mod["Thalach"]<=180)&(df_mod["Thalach"]>170),9,df_mod["Thalach"])
df_mod["Thalach"] = np.where((df_mod["Thalach"]<=190)&(df_mod["Thalach"]>180),10,df_mod["Thalach"])
df_mod["Thalach"] = np.where((df_mod["Thalach"]<=2000)&(df_mod["Thalach"]>190),11,df_mod["Thalach"])





df_usar = df_mod[["Age","Sex","Trestbps","Chol","Fbs","Restecg","Num","Thalach","CP","Exang"]]
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


################################################################################################
#-----------------------------------FRONT DE LA APLICACIÓN-------------------------------------#
###############################################################################################


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
                html.P("DISCLAIMER: El presente proyecto no representa a la Universidad de los Andes ni sus intereses.", style={'color': 'red', 'text-align':'center', 'font-size':'150%'}, className="fst-italic")
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
            html.Div([
                dcc.Graph(
                        id='heatmap',
                        figure=mpc
                    )
            ], style={'margin':'30px'}),
            html.Br(),
            
            html.Div([
                
                dcc.Graph(
                        id='heatmap',
                        figure=fig
                    )
            ],style={'margin':'30px'}),
            
            html.Br(),
            
             html.Div([
                
                dcc.Graph(
                        id='heatmap',
                        figure=fig1
                    )
            ],style={'margin':'30px'})
            
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
    

