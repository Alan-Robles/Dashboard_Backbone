import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import dash
from dash import dcc, html

import sys
import os

# Agrega el directorio actual al path para poder importar
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import create_ageb_layout, register_ageb_callbacks

url_clinicas = "https://raw.githubusercontent.com/Alan-Robles/Dashboard_Backbone/main/data/clinicas_con_ageb_y_nse.csv"
url_enfermedades = "https://raw.githubusercontent.com/Alan-Robles/Dashboard_Backbone/main/data/Datos_por_municipio.csv"
url_ageb = "https://raw.githubusercontent.com/Alan-Robles/Dashboard_Backbone/main/data/AGEB_NSE_NL.geojson"

clinicas_backbone = pd.read_csv(url_clinicas)
enfermedades_municipios = pd.read_csv(url_enfermedades)

agrupacion = {
    'Alto': 'Alto',
    
    'Medio-Alto': 'Medio-Alto',
    'Medio': 'Medio-Alto',
    
    'Medio-Bajo': 'Medio-Bajo',
    'Bajo-Alto': 'Medio-Bajo',
    
    'Bajo-Medio': 'Bajo',
    'Bajo': 'Bajo'
}

# Crear nueva columna o sobrescribir
clinicas_backbone['NSE_final'] = clinicas_backbone['NSE'].map(agrupacion)
merged = clinicas_backbone.merge(enfermedades_municipios, left_on=['State','Municipality'], right_on=['NOM_ENT','NOM_MUN'])

merged = merged.rename(columns={'Specialty of the \nMedical Equipment': 'Specialty of the Medical Equipment'})
merged = merged.rename(columns={'Number of \nConsultations per Month': 'Number of Consultations per Month'})

clinicas_backbone = clinicas_backbone.rename(columns={'Specialty of the \nMedical Equipment': 'Specialty of the Medical Equipment'})
clinicas_backbone = clinicas_backbone.rename(columns={'Number of \nConsultations per Month': 'Number of Consultations per Month'})



## Gráfico 1

clinic_counts = merged['Clinic Size'].value_counts().reset_index()
clinic_counts.columns = ['Clinic Size', 'Count']

# === Gráfico 1 ===
fig_1 = px.pie(
    clinic_counts,
    names='Clinic Size',
    values='Count',
    title='Distribución de tamaños de clínicas',
    color='Clinic Size',
    color_discrete_map={
        'Small': 'red',
        'Medium': 'blue',
        'Micro': 'orange',
    },
    hover_data=['Count']
)

fig_1.update_traces(
    textposition='inside',
    textinfo='percent+label',
    hoverinfo='label+percent'
)

# Gráfico 2
top20mun_consultations = (
    clinicas_backbone
    .groupby(['State', 'Municipality', 'Clinic Size'])['Number of Consultations per Month']
    .mean()
    .reset_index()
    .sort_values(by='Number of Consultations per Month', ascending=False)
    .head(20)
)

top20mun_consultations['estado_mun'] = top20mun_consultations['State'] + ', ' + top20mun_consultations['Municipality']

fig_2 = px.bar(
    top20mun_consultations,
    x='estado_mun',
    y='Number of Consultations per Month',
    color='Clinic Size',
    color_discrete_sequence=px.colors.sequential.Viridis,
    title='Top 20 Promedio de Consultas por Municipio y Tamaño de Clínica',
    hover_data={
        'Number of Consultations per Month': ':.2f',
        'estado_mun': True,
        'Clinic Size': True
    }
)

fig_2.update_layout(
    xaxis_title='Municipio',
    yaxis_title='Promedio de Consultas por Mes',
    xaxis_tickangle=-45,
    legend_title_text='Tamaño de Clínica',
    title_font_size=16,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12
)

# === FIGURA 3: MAPA 1 ===
fig_3 = px.scatter_mapbox(
    clinicas_backbone,
    lat='latitud',
    lon='longitud',
    hover_name='Clinic Aid Code',
    hover_data={
        'Clinic Size': True,
        'Number of Consultations per Month': ':.2f'
    },
    color='Clinic Size',
    size='Number of Consultations per Month',
    color_discrete_sequence=px.colors.sequential.Viridis,
    mapbox_style='open-street-map',
    zoom=4,
    height=600,
    title='Mapa de Clínicas Backbone por # Consultas por Mes'
)

fig_3.update_layout(
    title_font_size=18,
    margin=dict(l=20, r=20, t=60, b=20),
    mapbox=dict(center=dict(lat=23.6345, lon=-102.5528)),  # Centrado en México
)

# === FIGURA 4: MAPA DE COBERTURA VULNERABLE ===
merged['cobertura_pct'] = merged['Number of Consultations per Month'] / (merged['PSINDER'] + 1)

fig_4 = px.scatter_mapbox(
    merged,
    lat='latitud',
    lon='longitud',
    color='cobertura_pct',
    size='Number of Consultations per Month',
    mapbox_style='open-street-map',
    zoom=4,
    height=600,
    title='% de Cobertura Vulnerable Estimada por Clínica'
)

fig_4.update_layout(
    title_font_size=18,
    margin=dict(l=20, r=20, t=60, b=20),
    mapbox=dict(center=dict(lat=23.6345, lon=-102.5528)),
)

# === FIGURA 5: % PROMEDIO DE PACIENTES DE BAJO INGRESO POR TIPO DE CLÍNICA ===
grouped = clinicas_backbone.groupby('Clinic Size')['% of Patients with Middle-Low to Low Income'].mean().reset_index()

fig_5 = px.bar(
    grouped,
    x='Clinic Size',
    y='% of Patients with Middle-Low to Low Income',
    color='Clinic Size',
    color_discrete_sequence=['seagreen'],
    title='% Promedio de Pacientes con Ingresos Medio-Bajo o Bajo por Tipo de Clínica'
)

fig_5.update_layout(
    xaxis_title='Tamaño de Clínica',
    yaxis_title='Porcentaje (%)',
    title_font_size=16,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12,
    showlegend=False
)

# === FIGURA 6: MAPA DE CLÍNICAS POR % DE PACIENTES DE BAJO INGRESO ===
fig_6 = px.scatter_mapbox(
    clinicas_backbone,
    lat='latitud',
    lon='longitud',
    hover_name='Clinic Aid Code',
    color='Clinic Size',
    size='% of Patients with Middle-Low to Low Income',
    mapbox_style='open-street-map',
    zoom=4,
    height=600,
    title='Mapa de Clínicas Backbone por % de Pacientes de Bajos Ingresos'
)

fig_6.update_layout(
    title_font_size=18,
    margin=dict(l=20, r=20, t=60, b=20),
    mapbox=dict(center=dict(lat=23.6345, lon=-102.5528))
)

# === FIGURA 7: CLÍNICAS QUE ATIENDEN PACIENTES CON SEGURO MÉDICO ===
clinic_counts = clinicas_backbone['Attends Patients with Health Insurance'].value_counts().reset_index()
clinic_counts.columns = ['Atiende Seguro Médico', 'Cantidad']

fig_7 = px.pie(
    clinic_counts,
    names='Atiende Seguro Médico',
    values='Cantidad',
    color_discrete_sequence=px.colors.sequential.Viridis,
    title='Porcentaje de Clínicas que Atienden Pacientes con Seguro Médico'
)

fig_7.update_traces(textinfo='percent+label', pull=[0.05, 0])
fig_7.update_layout(title_font_size=16)

fig_8 = px.histogram(
    clinicas_backbone,
    x='Average Age of the SME (years)',
    nbins=20,
    title='Antigüedad de las Clínicas'
)
fig_8.update_traces(marker_color='indianred')
fig_8.update_layout(
    xaxis_title='Average Age of the SME (years)',
    yaxis_title='Frecuencia',
    title_font_size=16
)

fig_9 = px.box(
    clinicas_backbone,
    x='Specialty of the Medical Equipment',
    y='Average Age of the SME (years)',
    color='Specialty of the Medical Equipment',
    title='Especialidades con Clínicas con más antigüedad'
)
fig_9.update_layout(
    xaxis_title='Especialidad',
    yaxis_title='Promedio de Edad de SME',
    xaxis_tickangle=-45,
    title_font_size=16,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12,
    showlegend=False
)

fig_10 = px.scatter(
    merged,
    x='Average Age of the SME (years)',
    y='Number of Consultations per Month',
    color='Specialty of the Medical Equipment',
    size='% of Patients with Middle-Low to Low Income',
    hover_name='Specialty of the Medical Equipment',
    title='Edad vs Consultas por Especialidad (tamaño = % Pacientes vulnerables)',
    labels={
        'Average Age of the SME (years)': 'Años de operación',
        'Number of Consultations per Month': 'Consultas mensuales',
        '% of Patients with Middle-Low to Low Income': '% Pacientes bajos recursos'
    }
)

fig_11 = px.scatter(
    clinicas_backbone,
    x='Total Staff',
    y='Number of Consultations per Month',
    color='Clinic Size',
    size='Number of Consultations per Month',
    size_max=40,
    title='Relación entre Personal Médico y Consultas Mensuales'
)
fig_11.update_layout(
    xaxis_title='Personal Total',
    yaxis_title='Número de Consultas por Mes',
    title_font_size=16
)

clinicas_backbone['consultas_por_staff'] = clinicas_backbone['Number of Consultations per Month'] / clinicas_backbone['Total Staff']
fig_12 = px.box(
    clinicas_backbone,
    x='Specialty of the Medical Equipment',
    y='consultas_por_staff',
    color='Specialty of the Medical Equipment',
    title='Consultas por Personal por Especialidad del Equipo Médico'
)
fig_12.update_layout(
    xaxis_title='Especialidad del Equipo Médico',
    yaxis_title='Consultas por Personal',
    xaxis_tickangle=-45,
    title_font_size=16,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12,
    yaxis_showgrid=True,
    yaxis_gridcolor='lightgray',
    yaxis_gridwidth=0.5,
    showlegend=False
)

fig_13 = px.scatter_mapbox(
    clinicas_backbone,
    lat='latitud',
    lon='longitud',
    hover_name='Clinic Aid Code',
    color='NSE_final',
    mapbox_style='open-street-map',
    zoom=4,
    height=600,
    title='Mapa de las Clínicas Backbone por NSE'
)
fig_13.update_layout(
    title_font_size=18,
    margin=dict(l=20, r=20, t=60, b=20),
    mapbox=dict(center=dict(lat=23.6345, lon=-102.5528))
)

import numpy as np

# Preparar DataFrame con ratio
df = clinicas_backbone[['Clinic Aid Code', 'Clinic Size', 'Number of Consultations per Month', 'Total Staff', 'City', 'Specialty of the Medical Equipment']].copy()
df['operational_ratio'] = df['Number of Consultations per Month'] / df['Total Staff']
limits = {'Micro': 100, 'Small': 150, 'Medium': 200}
df['overloaded'] = df.apply(lambda x: x['operational_ratio'] > limits.get(x['Clinic Size'], np.inf), axis=1)
overloaded_clinics = df[df['overloaded']]

# fig_14 → Por City
city_counts = overloaded_clinics['City'].value_counts().head(10).reset_index()
city_counts.columns = ['City', 'Cantidad']

fig_14 = px.bar(
    city_counts,
    x='Cantidad',
    y='City',
    orientation='h',
    title='Clínicas Sobrecargadas por Ciudad',
    color='Cantidad',
    color_continuous_scale='Reds'
)
fig_14.update_layout(yaxis=dict(autorange="reversed"))  # Invertir eje y para barras horizontales

# fig_15 → Por Specialty
spec_counts = overloaded_clinics['Specialty of the Medical Equipment'].value_counts().head(10).reset_index()
spec_counts.columns = ['Especialidad', 'Cantidad']

fig_15 = px.bar(
    spec_counts,
    x='Cantidad',
    y='Especialidad',
    orientation='h',
    title='Clínicas Sobrecargadas por Especialidad',
    color='Cantidad',
    color_continuous_scale='Reds'
)
fig_15.update_layout(yaxis=dict(autorange="reversed"))

# === FIGURA 16: Distribución de NSE de las Clínicas ===
nse_counts = clinicas_backbone['NSE_final'].value_counts().reset_index()
nse_counts.columns = ['NSE_final', 'Cantidad']

fig_16 = px.pie(
    nse_counts,
    names='NSE_final',
    values='Cantidad',
    color_discrete_sequence=px.colors.sequential.Viridis,
    title='Distribución de NSE de las Clínicas'
)
fig_16.update_traces(textinfo='percent+label', pull=[0.05]*len(nse_counts))
fig_16.update_layout(title_font_size=16)

# Paso 1: contar cuántas clínicas hay por especialidad
counts = merged['Specialty of the Medical Equipment'].value_counts().reset_index()
counts.columns = ['Specialty of the Medical Equipment', 'Count']

# Paso 2: calcular porcentaje
counts['Percentage'] = 100 * counts['Count'] / counts['Count'].sum()

# Paso 3: agrupar las especialidades con <5% en "Others"
others = counts[counts['Percentage'] < 5]['Count'].sum()
main = counts[counts['Percentage'] >= 5].copy()

# agregar la categoría Others
if others > 0:
    main.loc[len(main)] = ['Others', others, 100 * others / counts['Count'].sum()]

# Paso 4: graficar
fig_17 = px.pie(
    main,
    names='Specialty of the Medical Equipment',
    values='Count',
    title='Distribución de especialidades de clínicas (agrupadas <5%)',
    color='Specialty of the Medical Equipment',
    hover_data=['Percentage']
)

fig_17.update_traces(
    textposition='inside',
    textinfo='percent+label',
    hoverinfo='label+percent+name'
)

# === FIGURA 18 ===
clinicas_backbone = clinicas_backbone.rename(columns={'Specialty of the Medical Equipment': 'Specialty of the Medical Equipment'})
fig_18 = px.scatter_mapbox(
    clinicas_backbone,
    lat='latitud',
    lon='longitud',
    hover_name='Clinic Aid Code',
    hover_data={
        'Specialty of the Medical Equipment': True,
    },
    color='Specialty of the Medical Equipment',
    color_discrete_sequence=px.colors.sequential.Viridis,
    mapbox_style='open-street-map',
    zoom=4,
    height=600,
    title='Mapa de Clínicas Backbone por # Consultas por Mes'
)

fig_18.update_layout(
    title_font_size=18,
    margin=dict(l=20, r=20, t=60, b=20),
    mapbox=dict(center=dict(lat=23.6345, lon=-102.5528)),  # Centrado en México
)

fig_18.update_traces(
    marker=dict(size=12),
    selector=dict(mode='markers')
)

# Paso 1: calcular promedio de consultas por especialidad
promedios = (
    clinicas_backbone
    .groupby('Specialty of the Medical Equipment', as_index=False)
    ['Number of Consultations per Month']
    .mean()
)

# Paso 2: graficar los promedios
fig_19 = px.bar(
    promedios,
    x='Specialty of the Medical Equipment',
    y='Number of Consultations per Month',
    title='Promedio de Consultas por Especialidad',
    hover_data={
        'Number of Consultations per Month': ':.2f',
    },
    color='Specialty of the Medical Equipment',  # opcional: colorear por especialidad
)

# Paso 3: personalizar diseño
fig_19.update_layout(
    xaxis_title='Especialidad',
    yaxis_title='Promedio de Consultas por Mes',
    xaxis_tickangle=-45,
    showlegend=False,  # ocultar leyenda si no es necesaria
    title_font_size=16,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12
)

fig_20 = px.bar(
    merged,
    x='NSE_final',
    y='Number of Consultations per Month',
    title='Total de Consultas por NSE',
    hover_data={
        'Number of Consultations per Month': ':.2f',
    },
    color='NSE_final',  # opcional: colorear por especialidad
)

fig_20.update_layout(
    xaxis_title='NSE',
    yaxis_title='Total de Consultas por Mes',
    xaxis_tickangle=-45,
    showlegend=False,  # ocultar leyenda si no es necesaria
    title_font_size=16,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12
)

# # ------------------ Figuras a json---------

# import plotly.io as pio
# import re
# import unidecode  # asegúrate de tenerlo instalado con: pip install unidecode

# # Lista con todas las figuras
# figs = [
#     fig_1, fig_2, fig_3, fig_4, fig_5, fig_6,
#     fig_7, fig_8, fig_9, fig_10, fig_11, fig_12,
#     fig_13, fig_14, fig_15, fig_16, fig_17, fig_18, fig_19, fig_20
# ]

# # Función para limpiar el título y crear un nombre de archivo seguro
# def clean_title(title):
#     # Eliminar acentos, signos raros y dejar solo letras, números y guiones bajos
#     title = unidecode.unidecode(title)
#     title = re.sub(r'[^A-Za-z0-9 _-]', '', title)
#     title = title.replace(' ', '_')
#     return title.strip('_')

# # Guardar cada figura con su título como nombre
# for fig in figs:
#     title = fig.layout.title.text or "grafico_sin_titulo"
#     filename = clean_title(title) + ".json"
#     pio.write_json(fig, filename)
#     print(f"✅ Guardado: {filename}")

# # === CREAR CARPETA DE EXPORTACIÓN ===
# import os
# output_folder = "json_datasets"
# os.makedirs(output_folder, exist_ok=True)

# # === EXPORTAR A JSON ===
# datasets = {
#     "clinicas_backbone.json": clinicas_backbone,
#     "enfermedades_municipios.json": enfermedades_municipios,
#     "merged.json": merged
# }

# for filename, df in datasets.items():
#     output_path = os.path.join(output_folder, filename)
#     df.to_json(output_path, orient='records', indent=4, force_ascii=False)
#     print(f"✅ Guardado: {output_path}")


# =========================================================
# APP DASH - Layout centrado y ancho completo
# =========================================================
from dash import Dash, dcc, html, Input, Output
from flask_caching import Cache
import time  # solo para simular carga lenta

import pandas as pd

# ==========================
# Inicialización de Dash
# ==========================
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Configurar caché
cache = Cache(app.server, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 600
})

# ==========================
# DataFrame de ejemplo
# ==========================
# merged = pd.read_csv("tu_archivo.csv")
# Asegúrate de que 'merged' contenga la columna 'State'

# Lista de estados para el filtro global
state_options = [{'label': s, 'value': s} for s in sorted(merged['State'].unique())]
state_options.insert(0, {'label': 'Todos', 'value': 'Todos'})  # opción para mostrar todo

# ==========================
# Funciones para generar gráficos filtrados
# ==========================
def generate_fig1(df):
    return fig_1  

def generate_fig2(df):
    return fig_2

def generate_fig3(df):
    return fig_3

def generate_fig4(df):
    return fig_4

def generate_fig5(df):
    return fig_5

def generate_fig6(df):
    return fig_6

def generate_fig7(df):
    return fig_7

def generate_fig8(df):
    return fig_8

def generate_fig9(df):
    return fig_9

def generate_fig10(df):
    return fig_10

def generate_fig11(df):
    return fig_11

def generate_fig12(df):
    return fig_12

def generate_fig14(df):
    return fig_14

def generate_fig15(df):
    return fig_15

def generate_fig13(df):
    return fig_13

def generate_fig16(df):
    return fig_16

def generate_fig16(df):
    return fig_17

def generate_fig16(df):
    return fig_18

def generate_fig16(df):
    return fig_19

def generate_fig16(df):
    return fig_20

# ==========================
# Crear dashboards
# ==========================
@cache.memoize()

@cache.memoize()
def create_dashboards(estado_filtro=None, nse_filtro=None, especialidad_filtro=None):
    # Aplicar filtro a los DataFrames principales
    if estado_filtro and estado_filtro != "Todos":
        clinicas_filtradas = clinicas_backbone[clinicas_backbone['State'] == estado_filtro].copy()
        merged_filtrado = merged[merged['State'] == estado_filtro].copy()
    else:
        clinicas_filtradas = clinicas_backbone.copy()
        merged_filtrado = merged.copy()
    
    # Aplicar filtro por NSE
    if nse_filtro and nse_filtro != "Todos":
        clinicas_filtradas = clinicas_filtradas[clinicas_filtradas['NSE_final'] == nse_filtro].copy()
        merged_filtrado = merged_filtrado[merged_filtrado['NSE_final'] == nse_filtro].copy()
    
    # Aplicar filtro por especialidad
    if especialidad_filtro and especialidad_filtro != "Todos":
        clinicas_filtradas = clinicas_filtradas[clinicas_filtradas['Specialty of the Medical Equipment'] == especialidad_filtro].copy()
        merged_filtrado = merged_filtrado[merged_filtrado['Specialty of the Medical Equipment'] == especialidad_filtro].copy()
    
    # Resto del código de creación de figuras permanece igual...
    # === Gráfico 1 - Actualizado para usar DataFrame filtrado ===
    clinic_counts = merged_filtrado['Clinic Size'].value_counts().reset_index()
    
    # === Gráfico 1 - Actualizado para usar DataFrame filtrado ===
    clinic_counts = merged_filtrado['Clinic Size'].value_counts().reset_index()
    clinic_counts.columns = ['Clinic Size', 'Count']

    fig_1 = px.pie(
        clinic_counts,
        names='Clinic Size',
        values='Count',
        title=f'Distribución de tamaños de clínicas{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}',
        color='Clinic Size',
        color_discrete_map={
            'Small': 'red',
            'Medium': 'blue',
            'Micro': 'orange',
        },
        hover_data=['Count']
    )
    fig_1.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+percent'
    )

    # === Gráfico 2 - Actualizado ===
    top20mun_consultations = (
        clinicas_filtradas
        .groupby(['State', 'Municipality', 'Clinic Size'])['Number of Consultations per Month']
        .mean()
        .reset_index()
        .sort_values(by='Number of Consultations per Month', ascending=False)
        .head(20)
    )

    top20mun_consultations['estado_mun'] = top20mun_consultations['State'] + ', ' + top20mun_consultations['Municipality']

    fig_2 = px.bar(
        top20mun_consultations,
        x='estado_mun',
        y='Number of Consultations per Month',
        color='Clinic Size',
        color_discrete_sequence=px.colors.sequential.Viridis,
        title=f'Top 20 Promedio de Consultas por Municipio y Tamaño de Clínica{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}',
        hover_data={
            'Number of Consultations per Month': ':.2f',
            'estado_mun': True,
            'Clinic Size': True
        }
    )
    fig_2.update_layout(
        xaxis_title='Municipio',
        yaxis_title='Promedio de Consultas por Mes',
        xaxis_tickangle=-45,
        legend_title_text='Tamaño de Clínica',
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12
    )

    # === FIGURA 3: MAPA 1 - Actualizado ===
    fig_3 = px.scatter_mapbox(
        clinicas_filtradas,
        lat='latitud',
        lon='longitud',
        hover_name='Clinic Aid Code',
        hover_data={
            'Clinic Size': True,
            'Number of Consultations per Month': ':.2f'
        },
        color='Clinic Size',
        size='Number of Consultations per Month',
        color_discrete_sequence=px.colors.sequential.Viridis,
        mapbox_style='open-street-map',
        zoom=4,
        height=600,
        title=f'Mapa de Clínicas Backbone por # Consultas por Mes{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )
    fig_3.update_layout(
        title_font_size=18,
        margin=dict(l=20, r=20, t=60, b=20),
        mapbox=dict(center=dict(lat=23.6345, lon=-102.5528)),
    )

    # === FIGURA 4: MAPA DE COBERTURA VULNERABLE - Actualizado ===
    merged_filtrado['cobertura_pct'] = merged_filtrado['Number of Consultations per Month'] / (merged_filtrado['PSINDER'] + 1)

    fig_4 = px.scatter_mapbox(
        merged_filtrado,
        lat='latitud',
        lon='longitud',
        color='cobertura_pct',
        size='Number of Consultations per Month',
        mapbox_style='open-street-map',
        zoom=4,
        height=600,
        title=f'% de Cobertura Vulnerable Estimada por Clínica{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )
    fig_4.update_layout(
        title_font_size=18,
        margin=dict(l=20, r=20, t=60, b=20),
        mapbox=dict(center=dict(lat=23.6345, lon=-102.5528)),
    )

    # === FIGURA 5: % PROMEDIO DE PACIENTES DE BAJO INGRESO - Actualizado ===
    grouped = clinicas_filtradas.groupby('Clinic Size')['% of Patients with Middle-Low to Low Income'].mean().reset_index()

    fig_5 = px.bar(
        grouped,
        x='Clinic Size',
        y='% of Patients with Middle-Low to Low Income',
        color='Clinic Size',
        color_discrete_sequence=['seagreen'],
        title=f'% Promedio de Pacientes con Ingresos Medio-Bajo o Bajo por Tipo de Clínica{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )
    fig_5.update_layout(
        xaxis_title='Tamaño de Clínica',
        yaxis_title='Porcentaje (%)',
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12,
        showlegend=False
    )

    # === FIGURA 6: MAPA DE CLÍNICAS POR % DE PACIENTES DE BAJO INGRESO - Actualizado ===
    fig_6 = px.scatter_mapbox(
        clinicas_filtradas,
        lat='latitud',
        lon='longitud',
        hover_name='Clinic Aid Code',
        color='Clinic Size',
        size='% of Patients with Middle-Low to Low Income',
        mapbox_style='open-street-map',
        zoom=4,
        height=600,
        title=f'Mapa de Clínicas Backbone por % de Pacientes de Bajos Ingresos{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )
    fig_6.update_layout(
        title_font_size=18,
        margin=dict(l=20, r=20, t=60, b=20),
        mapbox=dict(center=dict(lat=23.6345, lon=-102.5528))
    )

    # === FIGURA 7: CLÍNICAS QUE ATIENDEN PACIENTES CON SEGURO MÉDICO - Actualizado ===
    clinic_counts = clinicas_filtradas['Attends Patients with Health Insurance'].value_counts().reset_index()
    clinic_counts.columns = ['Atiende Seguro Médico', 'Cantidad']

    fig_7 = px.pie(
        clinic_counts,
        names='Atiende Seguro Médico',
        values='Cantidad',
        color_discrete_sequence=px.colors.sequential.Viridis,
        title=f'Porcentaje de Clínicas que Atienden Pacientes con Seguro Médico{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )
    fig_7.update_traces(textinfo='percent+label', pull=[0.05, 0])
    fig_7.update_layout(title_font_size=16)

    # === FIGURA 8: HISTOGRAMA ANTIGÜEDAD - Actualizado ===
    fig_8 = px.histogram(
        clinicas_filtradas,
        x='Average Age of the SME (years)',
        nbins=20,
        title=f'Antigüedad de las Clínicas{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )
    fig_8.update_traces(marker_color='indianred')
    fig_8.update_layout(
        xaxis_title='Average Age of the SME (years)',
        yaxis_title='Frecuencia',
        title_font_size=16
    )

    # === FIGURA 9: BOXPLOT ESPECIALIDADES - Actualizado ===
    fig_9 = px.box(
        clinicas_filtradas,
        x='Specialty of the Medical Equipment',
        y='Average Age of the SME (years)',
        color='Specialty of the Medical Equipment',
        title=f'Especialidades con Clínicas con más antigüedad{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )
    fig_9.update_layout(
        xaxis_title='Especialidad',
        yaxis_title='Promedio de Edad de SME',
        xaxis_tickangle=-45,
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12,
        showlegend=False
    )

    # === FIGURA 10: SCATTER EDAD VS CONSULTAS - Actualizado ===
    fig_10 = px.scatter(
        merged_filtrado,
        x='Average Age of the SME (years)',
        y='Number of Consultations per Month',
        color='Specialty of the Medical Equipment',
        size='% of Patients with Middle-Low to Low Income',
        hover_name='Specialty of the Medical Equipment',
        title=f'Edad vs Consultas por Especialidad{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}',
        labels={
            'Average Age of the SME (years)': 'Años de operación',
            'Number of Consultations per Month': 'Consultas mensuales',
            '% of Patients with Middle-Low to Low Income': '% Pacientes bajos recursos'
        }
    )

    # === FIGURA 11: SCATTER PERSONAL VS CONSULTAS - Actualizado ===
    fig_11 = px.scatter(
        clinicas_filtradas,
        x='Total Staff',
        y='Number of Consultations per Month',
        color='Clinic Size',
        size='Number of Consultations per Month',
        size_max=40,
        title=f'Relación entre Personal Médico y Consultas Mensuales{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )
    fig_11.update_layout(
        xaxis_title='Personal Total',
        yaxis_title='Número de Consultas por Mes',
        title_font_size=16
    )

    # === FIGURA 12: BOX CONSULTAS POR PERSONAL - Actualizado ===
    clinicas_filtradas['consultas_por_staff'] = clinicas_filtradas['Number of Consultations per Month'] / clinicas_filtradas['Total Staff']

    fig_12 = px.box(
        clinicas_filtradas,
        x='Specialty of the Medical Equipment',
        y='consultas_por_staff',
        color='Specialty of the Medical Equipment',
        title=f'Consultas por Personal por Especialidad del Equipo Médico{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )
    fig_12.update_layout(
        xaxis_title='Especialidad del Equipo Médico',
        yaxis_title='Consultas por Personal',
        xaxis_tickangle=-45,
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12,
        yaxis_showgrid=True,
        yaxis_gridcolor='lightgray',
        yaxis_gridwidth=0.5,
        showlegend=False
    )

    # === FIGURA 13: MAPA NSE - Actualizado ===
    fig_13 = px.scatter_mapbox(
        clinicas_filtradas,
        lat='latitud',
        lon='longitud',
        hover_name='Clinic Aid Code',
        color='NSE_final',
        size='% of Patients with Middle-Low to Low Income',
        mapbox_style='open-street-map',
        zoom=4,
        height=600,
        title=f'Mapa de Clínicas Backbone por NSE{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )
    fig_13.update_layout(
        title_font_size=18,
        margin=dict(l=20, r=20, t=60, b=20),
        mapbox=dict(center=dict(lat=23.6345, lon=-102.5528))
    )

    # === FIGURAS 14-15: CLÍNICAS SOBRECARGADAS - Actualizado ===
    df = clinicas_filtradas[['Clinic Aid Code', 'Clinic Size', 'Number of Consultations per Month', 'Total Staff', 'City', 'Specialty of the Medical Equipment']].copy()
    df['operational_ratio'] = df['Number of Consultations per Month'] / df['Total Staff']
    limits = {'Micro': 100, 'Small': 150, 'Medium': 200}
    df['overloaded'] = df.apply(lambda x: x['operational_ratio'] > limits.get(x['Clinic Size'], np.inf), axis=1)
    overloaded_clinics = df[df['overloaded']]

    # fig_14 → Por City
    city_counts = overloaded_clinics['City'].value_counts().head(10).reset_index()
    city_counts.columns = ['City', 'Cantidad']

    fig_14 = px.bar(
        city_counts,
        x='Cantidad',
        y='City',
        orientation='h',
        title=f'Clínicas Sobrecargadas por Ciudad{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}',
        color='Cantidad',
        color_continuous_scale='Reds'
    )
    fig_14.update_layout(yaxis=dict(autorange="reversed"))

    # fig_15 → Por Specialty
    spec_counts = overloaded_clinics['Specialty of the Medical Equipment'].value_counts().head(10).reset_index()
    spec_counts.columns = ['Especialidad', 'Cantidad']

    fig_15 = px.bar(
        spec_counts,
        x='Cantidad',
        y='Especialidad',
        orientation='h',
        title=f'Clínicas Sobrecargadas por Especialidad{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}',
        color='Cantidad',
        color_continuous_scale='Reds'
    )
    fig_15.update_layout(yaxis=dict(autorange="reversed"))

    # === FIGURA 16: Distribución de NSE - Actualizado ===
    nse_counts = clinicas_filtradas['NSE_final'].value_counts().reset_index()
    nse_counts.columns = ['NSE_final', 'Cantidad']

    fig_16 = px.pie(
        nse_counts,
        names='NSE_final',
        values='Cantidad',
        color_discrete_sequence=px.colors.sequential.Viridis,
        title=f'Distribución de NSE de las Clínicas{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )
    fig_16.update_traces(textinfo='percent+label', pull=[0.05]*len(nse_counts))
    fig_16.update_layout(title_font_size=16)

    # === FIGURA 17: Distribución de especialidades (agrupadas <5%) ===
    # Paso 1: contar cuántas clínicas hay por especialidad
    counts = merged_filtrado['Specialty of the Medical Equipment'].value_counts().reset_index()
    counts.columns = ['Specialty of the Medical Equipment', 'Count']

    # Paso 2: calcular porcentaje
    counts['Percentage'] = 100 * counts['Count'] / counts['Count'].sum()

    # Paso 3: agrupar las especialidades con <5% en "Others"
    others = counts[counts['Percentage'] < 5]['Count'].sum()
    main = counts[counts['Percentage'] >= 5].copy()

    # agregar la categoría Others
    if others > 0:
        main.loc[len(main)] = ['Others', others, 100 * others / counts['Count'].sum()]

    # Paso 4: graficar
    fig_17 = px.pie(
        main,
        names='Specialty of the Medical Equipment',
        values='Count',
        title=f'Distribución de especialidades de clínicas (agrupadas <5%){" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}',
        color='Specialty of the Medical Equipment',
        hover_data=['Percentage']
    )

    fig_17.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+percent+name'
    )

    # === FIGURA 18: Mapa de clínicas por especialidad ===
    fig_18 = px.scatter_mapbox(
        clinicas_filtradas,
        lat='latitud',
        lon='longitud',
        hover_name='Clinic Aid Code',
        hover_data={
            'Specialty of the Medical Equipment': True,
            'Clinic Size': True,
            'Number of Consultations per Month': True
        },
        color='Specialty of the Medical Equipment',
        color_discrete_sequence=px.colors.sequential.Viridis,
        mapbox_style='open-street-map',
        zoom=4,
        height=600,
        title=f'Mapa de Clínicas Backbone por Especialidad{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )

    fig_18.update_layout(
        title_font_size=18,
        margin=dict(l=20, r=20, t=60, b=20),
        mapbox=dict(center=dict(lat=23.6345, lon=-102.5528)),
    )

    fig_18.update_traces(
        marker=dict(size=12),
        selector=dict(mode='markers')
    )

    # === FIGURA 19: Promedio de consultas por especialidad ===
    # Paso 1: calcular promedio de consultas por especialidad
    promedios = (
        clinicas_filtradas
        .groupby('Specialty of the Medical Equipment', as_index=False)
        ['Number of Consultations per Month']
        .mean()
        .sort_values('Number of Consultations per Month', ascending=False)
    )

    # Paso 2: graficar los promedios
    fig_19 = px.bar(
        promedios,
        x='Specialty of the Medical Equipment',
        y='Number of Consultations per Month',
        title=f'Promedio de Consultas por Especialidad{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}',
        hover_data={
            'Number of Consultations per Month': ':.2f',
        },
        color='Specialty of the Medical Equipment',
    )

    # Paso 3: personalizar diseño
    fig_19.update_layout(
        xaxis_title='Especialidad',
        yaxis_title='Promedio de Consultas por Mes',
        xaxis_tickangle=-45,
        showlegend=False,
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12
    )

    # === FIGURA 20: Total de consultas por NSE ===
    # Agrupar por NSE y sumar las consultas
    nse_consultas = (
        merged_filtrado
        .groupby('NSE_final', as_index=False)
        ['Number of Consultations per Month']
        .sum()
        .sort_values('Number of Consultations per Month', ascending=False)
    )

    fig_20 = px.bar(
        nse_consultas,
        x='NSE_final',
        y='Number of Consultations per Month',
        title=f'Total de Consultas por NSE{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}',
        hover_data={
            'Number of Consultations per Month': ':.2f',
        },
        color='NSE_final',
    )

    fig_20.update_layout(
        xaxis_title='NSE',
        yaxis_title='Total de Consultas por Mes',
        xaxis_tickangle=-45,
        showlegend=False,
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12
    )
    # === DASHBOARDS REORGANIZADOS ===
    
    # Dashboard 1: fig_13, fig_16, fig_17
    dashboard1 = html.Div([
        html.H2(
            f"Clínicas Backbone{' - ' + estado_filtro if estado_filtro and estado_filtro != 'Todos' else ''}",
            style={'textAlign': 'center'}
        ),
        html.Div([
            dcc.Graph(figure=fig_13, style={'gridArea': 'fig13', 'height': '400px'}, config={'responsive': True}),
            dcc.Graph(figure=fig_16, style={'gridArea': 'fig16', 'height': '400px'}, config={'responsive': True}),
            dcc.Graph(figure=fig_17, style={'gridArea': 'fig17', 'height': '500px'}, config={'responsive': True}),
        ], style={
            'display': 'grid',
            'gridTemplateColumns': '1fr 1fr',
            'gridTemplateRows': 'auto auto',
            'gridTemplateAreas': '''
                "fig16 fig17"
                "fig13 fig13"
            ''',
            'gap': '10px',
            'justifyItems': 'stretch',
            'alignItems': 'stretch'
        })
    ])

    # Dashboard 2: fig_3, fig_18, fig_19, fig_20
    dashboard2 = html.Div([
        html.H2(
            f"Población beneficiada{' - ' + estado_filtro if estado_filtro and estado_filtro != 'Todos' else ''}",
            style={'textAlign': 'center'}
        ),
        html.Div([
            dcc.Graph(figure=fig_3, style={'gridArea': 'fig3', 'height': '500px'}, config={'responsive': True}),
            dcc.Graph(figure=fig_18, style={'gridArea': 'fig18', 'height': '400px'}, config={'responsive': True}),
            dcc.Graph(figure=fig_19, style={'gridArea': 'fig19', 'height': '400px'}, config={'responsive': True}),
            dcc.Graph(figure=fig_20, style={'gridArea': 'fig20', 'height': '500px'}, config={'responsive': True}),
        ], style={
            'display': 'grid',
            'gridTemplateColumns': '1fr 1fr',
            'gridTemplateRows': 'auto auto',
            'gridTemplateAreas': '''
                "fig20 fig19"
                "fig3 fig3"
                "fig18 fig18"
            ''',
            'gap': '10px',
            'justifyItems': 'stretch'
        })
    ])

    # Dashboard 3: fig_1, fig_4, fig_8, fig_9
    dashboard3 = html.Div([
        html.H2(
            f"Perfil general de las clínicas{' - ' + estado_filtro if estado_filtro and estado_filtro != 'Todos' else ''}",
            style={'textAlign': 'center'}
        ),
        html.Div([
            dcc.Graph(figure=fig_1, style={'gridArea': 'fig1', 'height': '400px'}, config={'responsive': True}),
            dcc.Graph(figure=fig_4, style={'gridArea': 'fig4', 'height': '400px'}, config={'responsive': True}),
            dcc.Graph(figure=fig_8, style={'gridArea': 'fig8', 'height': '400px'}, config={'responsive': True}),
            dcc.Graph(figure=fig_9, style={'gridArea': 'fig9', 'height': '400px'}, config={'responsive': True}),
        ], style={
            'display': 'grid',
            'gridTemplateColumns': '1fr 1fr 1fr',
            'gridTemplateRows': 'auto auto',
            'gridTemplateAreas': '''
                "fig1 fig8 fig9"
                "fig4 fig4 fig4"
            ''',
            'gap': '10px',
            'justifyItems': 'stretch'
        })
    ])

    # Dashboard 4: fig_11, fig_12, fig_15, fig_14
    dashboard4 = html.Div([
        html.H2(
            f"Perfil operatorio de las clínicas{' - ' + estado_filtro if estado_filtro and estado_filtro != 'Todos' else ''}",
            style={'textAlign': 'center'}
        ),
        html.Div([
            dcc.Graph(figure=fig_11, style={'gridArea': 'fig11', 'height': '400px'}, config={'responsive': True}),
            dcc.Graph(figure=fig_12, style={'gridArea': 'fig12', 'height': '400px'}, config={'responsive': True}),
            dcc.Graph(figure=fig_15, style={'gridArea': 'fig15', 'height': '500px'}, config={'responsive': True}),
            dcc.Graph(figure=fig_14, style={'gridArea': 'fig14', 'height': '400px'}, config={'responsive': True}),
        ], style={
            'display': 'grid',
            'gridTemplateColumns': '1fr 1fr',
            'gridTemplateRows': 'auto auto auto',
            'gridTemplateAreas': '''
                "fig11 fig12"
                "fig14 fig15"
            ''',
            'gap': '10px',
            'justifyItems': 'stretch'
        })
    ])

    # Dashboard 5: fig_7, fig_5, fig_6
    dashboard5 = html.Div([
        html.H2(
            f"Perfil de los pacientes{' - ' + estado_filtro if estado_filtro and estado_filtro != 'Todos' else ''}",
            style={'textAlign': 'center'}
        ),
        html.Div([
            dcc.Graph(figure=fig_7, style={'gridArea': 'fig7', 'height': '500px'}, config={'responsive': True}),
            dcc.Graph(figure=fig_5, style={'gridArea': 'fig5', 'height': '400px'}, config={'responsive': True}),
            dcc.Graph(figure=fig_6, style={'gridArea': 'fig6', 'height': '500px'}, config={'responsive': True}),
        ], style={
            'display': 'grid',
            'gridTemplateColumns': '1fr 1fr',
            'gridTemplateRows': 'auto auto',
            'gridTemplateAreas': '''
                "fig5 fig7"
                "fig6 fig6"
            ''',
            'gap': '10px',
            'justifyItems': 'stretch'
        })
    ])

    return {
        'dashboard1': dashboard1,
        'dashboard2': dashboard2,
        'dashboard3': dashboard3,
        'dashboard4': dashboard4,
        'dashboard5': dashboard5
    }

# ==========================
# Layout principal
# ==========================
# Obtener opciones únicas para los dropdowns
estados = ["Todos"] + sorted(clinicas_backbone['State'].dropna().unique().tolist())
nse_options = ["Todos"] + sorted(clinicas_backbone['NSE_final'].dropna().unique().tolist())
especialidades = ["Todos"] + sorted(clinicas_backbone['Specialty of the Medical Equipment'].dropna().unique().tolist())

# === Layout principal ===
app.layout = html.Div([
    # Header con título y filtros
    html.Div([
        # Título a la izquierda
        html.Div([
            html.H1("Análisis Integral de Clínicas Backbone", style={
                'margin': '0', 'fontFamily': 'Arial, sans-serif', 'color': '#333'
            })
        ], style={'flex': '1'}),
        
        # Filtros a la derecha
        html.Div([
            # Filtro por estado
            html.Div([
                html.Label("Estado:", style={
                    'fontWeight': 'bold', 
                    'marginRight': '5px',
                    'fontSize': '14px'
                }),
                dcc.Dropdown(
                    id='estado-filter',
                    options=[{'label': estado, 'value': estado} for estado in estados],
                    value='Todos',
                    style={'width': '200px', 'display': 'inline-block'},
                    clearable=False
                )
            ], style={'display': 'inline-block', 'marginRight': '20px'}),
            
            # Filtro por NSE
            html.Div([
                html.Label("NSE:", style={
                    'fontWeight': 'bold', 
                    'marginRight': '5px',
                    'fontSize': '14px'
                }),
                dcc.Dropdown(
                    id='nse-filter',
                    options=[{'label': nse, 'value': nse} for nse in nse_options],
                    value='Todos',
                    style={'width': '150px', 'display': 'inline-block'},
                    clearable=False
                )
            ], style={'display': 'inline-block', 'marginRight': '20px'}),
            
            # Filtro por especialidad
            html.Div([
                html.Label("Especialidad:", style={
                    'fontWeight': 'bold', 
                    'marginRight': '5px',
                    'fontSize': '14px'
                }),
                dcc.Dropdown(
                    id='especialidad-filter',
                    options=[{'label': esp, 'value': esp} for esp in especialidades],
                    value='Todos',
                    style={'width': '200px', 'display': 'inline-block'},
                    clearable=False
                )
            ], style={'display': 'inline-block'})
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-end'})
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'marginBottom': '20px',
        'padding': '10px',
        'backgroundColor': '#f8f9fa',
        'borderRadius': '5px'
    }),

    # En tu app.layout, modifica la sección de tabs:
    dcc.Tabs(id='tabs', value='dashboard1', children=[
    dcc.Tab(label='Dashboard 1 – Clínicas Backbone', value='dashboard1'),
    dcc.Tab(label='Dashboard 2 – Población beneficiada', value='dashboard2'),
    dcc.Tab(label='Dashboard 3 – Perfil general de las clínicas', value='dashboard3'),
    dcc.Tab(label='Dashboard 4 – Perfil operatorio de las clínicas', value='dashboard4'),
    dcc.Tab(label='Dashboard 5 – Perfil de los pacientes', value='dashboard5'),
    dcc.Tab(label='Dashboard 6 – Backbone APP', value='dashboard6'),  # NUEVA PESTAÑA
    ]),

    # Contenedor para los dashboards
    html.Div(id='dashboard-content')
])

# === Callback para actualizar los dashboards cuando cambian los filtros o la pestaña ===
@app.callback(
    Output('dashboard-content', 'children'),
    [Input('tabs', 'value'),
     Input('estado-filter', 'value'),
     Input('nse-filter', 'value'),
     Input('especialidad-filter', 'value')]
)
def update_dashboard(tab_value, estado_filtro, nse_filtro, especialidad_filtro):
    if tab_value == 'dashboard6':
        # Retorna la app de AGEB
        return create_ageb_layout()
    else:
        # Generar dashboards con los filtros aplicados
        dashboards = create_dashboards(estado_filtro, nse_filtro, especialidad_filtro)
        return dashboards.get(tab_value, dashboards['dashboard1'])

# Registra los callbacks de AGEB DESPUÉS de definir la app
register_ageb_callbacks(app)

# ==========================
# Ejecutar app
# ==========================
if __name__ == '__main__':
    app.run(debug=True)
