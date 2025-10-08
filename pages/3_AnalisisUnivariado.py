import streamlit as st
import pandas as pd
import plotly.express as px
from utils import analizar_categoricas,resumen_numericas,GraficarCategoricas,graficar_numericas

st.title("Analisis Univariado del dataset")
# Verificar si los datos están disponibles
if 'df_transformado' in st.session_state:
    df = st.session_state.df_transformado
    st.success("Datos cargados desde otra página.")
    st.dataframe(df.head())
else:
    st.error("No se encontraron datos transformados. Ve a la página anterior primero.")

opcion = st.sidebar.radio("Selecciona el análisis:", 
                          ["Resumen General", "Variables Categóricas", "Variables Numéricas", "Variable Individual"])

if opcion == "Resumen General":
    st.header("Resumen General del Dataset")

    st.subheader("Variables categóricas")
    resumen_cat = analizar_categoricas(df)
    st.dataframe(resumen_cat)

    st.subheader("Variables numéricas")
    resumen_num = resumen_numericas(df)
    if resumen_num is not None:
        st.dataframe(resumen_num)

elif opcion == "Variables Categóricas":
    st.header("Análisis y Visualización de Variables Categóricas")

    resumen_cat = analizar_categoricas(df)
    st.dataframe(resumen_cat)

    tipo_graf = st.radio("Tipo de gráfico", options=["barra", "pastel"])
    GraficarCategoricas(df, tipo=tipo_graf)

elif opcion == "Variables Numéricas":
    st.header("Análisis y Visualización de Variables Numéricas")

    resumen_num = resumen_numericas(df)
    if resumen_num is not None:
        st.dataframe(resumen_num)

    graficar_numericas(df)

if opcion == "Variable Individual":
    st.header("Análisis Univariado de una Variable")

    column = st.selectbox("Selecciona la variable para analizar", df.columns)
    st.write(f"### Variable seleccionada: `{column}`")

    if pd.api.types.is_numeric_dtype(df[column]):
        st.subheader(" Estadísticas descriptivas")
        st.dataframe(df[[column]].describe().T)

        st.write(f"**Mediana:** {df[column].median():.2f}")
        st.write(f"**Varianza:** {df[column].var():.2f}")

        st.subheader(" Histograma (Plotly)")
        fig = px.histogram(df, x=column, nbins=20, title=f"Histograma de {column}",
                           labels={column: column}, marginal="box")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.subheader(" Frecuencia de categorías")
        conteo = df[column].value_counts(dropna=False).reset_index()
        conteo.columns = [column, "Frecuencia"]
        st.dataframe(conteo)

        st.subheader(" Gráfico de barras")
        fig_bar = px.bar(conteo, x=column, y="Frecuencia", title=f"Frecuencia de {column}",
                         text="Frecuencia", labels={column: "Categoría"})
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader(" Gráfico de pastel")
        if conteo.shape[0] > 6:
            st.warning(f"La variable `{column}` tiene más de 6 categorías. El gráfico de pastel no se mostrará.")
        else:
            fig_pie = px.pie(conteo, values="Frecuencia", names=column,
                             title=f"Distribución de {column}")
            st.plotly_chart(fig_pie, use_container_width=True)
