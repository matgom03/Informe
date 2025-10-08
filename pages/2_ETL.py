import streamlit as st
import pandas as pd
from utils import load_data,Summary

st.title("ETL - Extracción, Transformación y Carga, Analisis inicial")
# ========================
# Extraccion de los datos
# ========================
st.markdown("Se realizo la concatenacion de los dos dataframes el de train y test para crear un dataset general, agregando tambien los nombres de las columnas")

df1_head, df2_head, df = load_data()

st.subheader("Vista previa de 'adult.data'")

st.dataframe(df1_head)

st.subheader("Vista previa de 'adult.test'")

st.dataframe(df2_head)

st.divider()

st.subheader("Dataset final concatenado")

st.dataframe(df.head())

st.markdown("Ademas se realizo la transformacion de algunos de los tipos de datos")

# ========================
# Transformaciones de columnas
# ========================

st.divider()
st.subheader("Tipos de datos antes de transformación")
st.code(df.dtypes.astype(str))

# Transformar columnas
df["Education-num"] = df["Education-num"].astype("category")
df["Income"] = df["Income"].apply(lambda x: 1 if str(x).strip().startswith(">50K") else 0)
df["Income"] = df["Income"].astype("category")

st.subheader("Tipos de datos después de transformación")
st.code(df.dtypes.astype(str))
# ========================
# Summary de los datos
# ========================

st.markdown("Se realiza el summary inicial de los datos ")

Summary(df, "Datos cargados")

st.markdown("Revisamos las filas duplicadas del dataset")
duplicadas = df[df.duplicated()]

st.subheader("Filas duplicadas")

st.write(f"Total de filas duplicadas: {len(duplicadas)}")

if len(duplicadas) > 0:
    st.dataframe(duplicadas)
else:
    st.success("No se encontraron filas duplicadas.")

st.markdown("""En general, estas filas duplicadas se deben mas a casualidad que a error, son simplemente personas que tienen datos similares entre si, por lo que no se van a eliminar """)


st.session_state.df_transformado = df
st.write("Datos transformados:")
st.dataframe(df)


