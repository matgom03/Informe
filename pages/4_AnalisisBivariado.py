import streamlit as st
import pandas as pd
from utils import prueba_normalidad,analisis_bivariado_numerico,analisis_bivariado_categorico,analisis_bivariado_cat_num

st.title("Analisis bivariado del dataset")
if 'df_transformado' in st.session_state:
    df = st.session_state.df_transformado
    st.success("Datos cargados desde otra página.")
    st.dataframe(df.head())
else:
    st.error("No se encontraron datos transformados. Ve a la página anterior primero.")
st.markdown("se realizo el analisis bivariado del dataset entre las vairables numericas, categoricas y mixta")

st.markdown("antes de realizar el analisis se estudio la normalidad de las variables numericas para verificar si se deben usar algunas pruebas analiticas especificas")

prueba_normalidad(df)

st.markdown("""Podemos observar como al tener mas de 5000 instancias no podemos usar shapiro wilk por lo que se toma mejor los otros test como ks test y podemos observar que las variables numericas de nuestro dataset no son normales
            Tomaremos esto en cuenta a la hora de realizar el analisis bivariado""")
if df is not None:
    # Selección de tipo de análisis
    analisis = st.selectbox("Selecciona el tipo de análisis bivariado:", [
        "Numérico vs Numérico",
        "Categórico vs Categórico",
        "Numérico vs Categórico"
    ])

    top_n = st.slider("¿Cuántas relaciones mostrar (top)?", 3, 10, 5)
    mostrar_graficos = st.checkbox("¿Mostrar gráficos?", value=True)

    if analisis == "Numérico vs Numérico":
        st.header("Análisis Bivariado: Numérico vs Numérico")
        analisis_bivariado_numerico(df, top=top_n, mostrar_graficos=mostrar_graficos)

    elif analisis == "Categórico vs Categórico":
        st.header("Análisis Bivariado: Categórico vs Categórico")
        tipo_grafico = st.selectbox("Tipo de gráfico para proporciones:", ['heatmap'])
        analisis_bivariado_categorico(df, top=top_n, mostrar_graficos=mostrar_graficos, grafico=tipo_grafico)

    elif analisis == "Numérico vs Categórico":
        st.header("Análisis Bivariado: Numérico vs Categórico")
        analisis_bivariado_cat_num(df, mostrar_graficos=mostrar_graficos)





