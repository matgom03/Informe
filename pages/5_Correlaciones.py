import streamlit as st
import pandas as pd
from utils import analizar_colinealidad_y_correlaciones

st.title("Correlaciones y mutlicolinealidad del dataset")
# Verificar si el dataframe transformado está disponible
if 'df_transformado' in st.session_state:
    df = st.session_state.df_transformado
    st.success("Datos cargados desde session_state.")

    mostrar_graficos = st.checkbox("Mostrar gráficos", value=True)
    umbral_vif = st.slider("Umbral VIF", min_value=5, max_value=20, value=10)

    resultados = analizar_colinealidad_y_correlaciones(df, umbral_vif=umbral_vif, mostrar_graficos=mostrar_graficos)

else:
    st.warning("Primero ve a la página de transformación y guarda los datos.")