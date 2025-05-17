# save this as app.py
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import pearsonr
import streamlit as st

# Load Data
df = pd.read_csv("immunization_master_data.csv")

# Preprocess
siglas_estados = ['AC','AL','AM','AP','BA','CE','DF','ES','GO','MA','MG','MS','MT',
                  'PA','PB','PE','PI','PR','RJ','RN','RO','RR','RS','SC','SE','SP','TO']
estado_para_regiao = {
    'AC': 'North', 'AP': 'North', 'AM': 'North', 'PA': 'North', 'RO': 'North', 'RR': 'North', 'TO': 'North',
    'AL': 'Northeast', 'BA': 'Northeast', 'CE': 'Northeast', 'MA': 'Northeast', 'PB': 'Northeast',
    'PE': 'Northeast', 'PI': 'Northeast', 'RN': 'Northeast', 'SE': 'Northeast',
    'DF': 'Central-West', 'GO': 'Central-West', 'MT': 'Central-West', 'MS': 'Central-West',
    'ES': 'Southeast', 'MG': 'Southeast', 'RJ': 'Southeast', 'SP': 'Southeast',
    'PR': 'South', 'RS': 'South', 'SC': 'South'
}
df = df[df['LOCAL_NAME'].isin(siglas_estados)].copy()
df['region'] = df['LOCAL_NAME'].map(estado_para_regiao)

vacinas_basicas = [
    'FL_U1_BCG', 'FL_U1_POLIO', 'FL_U1_DTP', 'FL_U1_HepB',
    'FL_U1_Hib', 'FL_Y1_DTP', 'FL_Y1_POLIO', 'FL_Y1_MMR1'
]
df = df[
    (df['INDICATOR'].isin(vacinas_basicas)) &
    (df['AGE'].isin(['0-1 ano', '1 ano'])) &
    (df['PC_COVERAGE'].notna())
]

# Group by state and year
df_grouped = df.groupby(['LOCAL_NAME', 'region', 'YEAR']).agg({
    'PC_COVERAGE': 'mean',
    'MHDI_I': 'mean'
}).reset_index()

# Sidebar
st.sidebar.title("Controle")
selected_year = st.sidebar.slider("Ano", int(df_grouped['YEAR'].min()), int(df_grouped['YEAR'].max()), 2021)
show_regression = st.sidebar.checkbox("Mostrar linha de regressão", value=True)

# Filter for year
df_year = df_grouped[df_grouped['YEAR'] == selected_year]

# Correlation
corr, p_val = pearsonr(df_year['MHDI_I'], df_year['PC_COVERAGE'])
subtitle = f"Correlação de Pearson: {corr:.3f} | P-valor: {p_val:.3f}"

# Scatter Plot
show_labels = st.sidebar.checkbox("Mostrar nomes dos estados", value=False)

# Add text labels only if selected
fig = px.scatter(
    df_year,
    x='MHDI_I',
    y='PC_COVERAGE',
    color='region',
    hover_data=['LOCAL_NAME'],
    text='LOCAL_NAME' if show_labels else None,
    labels={
        'MHDI_I': 'Índice de Desenvolvimento Humano (MHDI_I)',
        'PC_COVERAGE': 'Cobertura Vacinal Média (%)'
    },
    title=f'Cobertura Vacinal vs MHDI_I — {selected_year}<br><sup>{subtitle}</sup>',
    height=600
)

if show_labels:
    fig.update_traces(textposition='top center')

# Add regression line
if show_regression:
    coef = np.polyfit(df_year['MHDI_I'], df_year['PC_COVERAGE'], 1)
    x_vals = np.linspace(df_year['MHDI_I'].min(), df_year['MHDI_I'].max(), 100)
    y_vals = coef[0] * x_vals + coef[1]
    fig.add_scatter(x=x_vals, y=y_vals, mode='lines', name='Regressão Linear')

# Show plot
st.plotly_chart(fig, use_container_width=True)
