import streamlit as st

st.set_page_config(page_title="EDA Adult Dataset", layout="wide", page_icon="📊")


st.title("Exploratory Data Analysis - Adult Dataset")


st.sidebar.success("Selecciona una sección desde la barra lateral")


st.markdown("""
Bienvenido al panel interactivo del **EDA (Análisis Exploratorio de Datos)**
para el dataset **Adult / Census Income** del repositorio UCI.


Usa la barra lateral para navegar entre las secciones del análisis.
- Contexto
- ETL
- Análisis Univariado
- Análisis Bivariado
- Correlaciones y Colinealidad
- Imputación
""")