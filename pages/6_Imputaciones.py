import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency
from utils import detectar_outliers,winsorizar_outliers,graficar_numericas,hot_deck_group,Cramers_v
st.title("Tratamiento de Outliers y Valores Faltantes")
if 'df_transformado' in st.session_state:
    df_original = st.session_state.df_transformado.copy()
    df_modificado = df_original.copy()

    st.subheader(" Variables con Outliers Detectados")
    outliers_df = detectar_outliers(df_original)
    st.dataframe(outliers_df)

    st.markdown("###  Aplicar Winsorización")
    aplicar_wins = st.checkbox("¿Aplicar winsorización?", value=False)

    if aplicar_wins:
        columnas_wins = st.multiselect("Selecciona variables para winsorizar:", 
                                       options=df_original.select_dtypes(include=[np.number]).columns.tolist(),
                                       default=outliers_df.head(3)['variable'].tolist())
        p_inf = st.slider("Percentil inferior", 0.0, 0.1, 0.01, step=0.01)
        p_sup = st.slider("Percentil superior", 0.9, 1.0, 0.99, step=0.01)

        df_modificado = winsorizar_outliers(df_modificado, columnas_wins, lower_percentile=p_inf, upper_percentile=p_sup)
        st.markdown("### Gráficas después de Winsorización")
        graficar_numericas(df_modificado)
    st.markdown("### Imputación de NAs")
    na_cols = df_modificado.columns[df_modificado.isna().sum() > 0].tolist()

    if na_cols:
        missing = df_modificado.isna().sum()
        missing_percent = (missing / len(df_modificado)) * 100
        missing_table = pd.DataFrame({
            'Valores_Faltantes': missing,
            'Porcentaje(%)': missing_percent.round(2)
        })
        missing_table = missing_table[missing_table['Valores_Faltantes'] > 0]

        st.subheader("Resumen de valores faltantes")
        st.dataframe(missing_table)
        
        st.markdown("### Imputación por moda (categóricas)")
        columnas_a_imputar = ['Native-country']
        df_filtrado = df_modificado.copy()  
        imp_median = SimpleImputer(strategy="most_frequent")
        df_imputado = df_filtrado.copy()
        df_imputado[columnas_a_imputar] = imp_median.fit_transform(df_imputado[columnas_a_imputar])
        for var in columnas_a_imputar:
            st.markdown(f"#### {var}: Comparación antes y después de imputación")

            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

            sns.countplot(x=df_filtrado[var], ax=axes[0], palette="pastel")
            axes[0].set_title(f"{var} - Antes de imputar")
            axes[0].set_ylabel("Frecuencia")
            axes[0].tick_params(axis='x', rotation=45)

            sns.countplot(x=df_imputado[var], ax=axes[1], palette="muted")
            axes[1].set_title(f"{var} - Después de imputar (moda)")
            axes[1].set_ylabel("Frecuencia")
            axes[1].tick_params(axis='x', rotation=45)

            plt.suptitle(f"Distribución de {var}", fontsize=14, weight="bold")
            plt.tight_layout()
            st.pyplot(fig)

            # Comparación estadística (Chi²)
            orig_counts = df_filtrado[var].value_counts().sort_index()
            imp_counts = df_imputado[var].value_counts().sort_index()

            # Alineamos los índices por si hay categorías que no existían antes
            orig_counts, imp_counts = orig_counts.align(imp_counts, fill_value=0)
            tabla = pd.DataFrame({"Original": orig_counts, "Imputada": imp_counts}).T

            # Chi-cuadrado
            chi2, p, dof, expected = chi2_contingency(tabla)

            # Mostrar resultados
            st.markdown("#####  Prueba Chi² para comparar proporciones")
            st.write("Proporciones antes de imputar:")
            st.write((orig_counts / orig_counts.sum()).round(3))

            st.write("Proporciones después de imputar:")
            st.write((imp_counts / imp_counts.sum()).round(3))

            st.write(f"**Chi² = {chi2:.3f}**, grados de libertad = {dof}, **p-valor = {p:.4f}**")

            if p > 0.05:
                st.success(" No se rechaza H₀: las distribuciones son estadísticamente iguales.")
            else:
                st.warning(" Se rechaza H₀: las distribuciones son diferentes tras imputar.")

            st.markdown("---")
        
        st.markdown("### Imputación por Hotdeck (categóricas)")
        df_filtrado["Workclass_hd"] = hot_deck_group(df_filtrado, "Workclass", "Education")
        df_filtrado["Occupation_hd"] = hot_deck_group(df_filtrado, "Occupation", "Native-country")


        df_filtrado["Workclass_hd"] = df_filtrado["Workclass_hd"].astype("category")
        df_filtrado["Occupation_hd"] = df_filtrado["Occupation_hd"].astype("category")

        df_filtrado["Workclass_hd"] = df_filtrado["Workclass_hd"].cat.set_categories(df_filtrado["Workclass"].dropna().unique())
        df_filtrado["Occupation_hd"] = df_filtrado["Occupation_hd"].cat.set_categories(df_filtrado["Occupation"].dropna().unique())
        variables = ["Workclass", "Occupation"]

        for var in variables:
  
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

            sns.countplot(x=df_filtrado[var], ax=axes[0], palette="pastel",
                            order=df_filtrado[var].dropna().unique())
            axes[0].set_title(f"{var} - Antes de imputar")
            axes[0].set_ylabel("Frecuencia")

            sns.countplot(x=df_filtrado[f"{var}_hd"], ax=axes[1], palette="muted",
                            order=df_filtrado[var].dropna().unique())
            axes[1].set_title(f"{var} - Después de imputar (Hot-deck)")
            axes[1].set_ylabel("Frecuencia")

            plt.suptitle(f"Comparación de distribución en {var}", fontsize=14, weight="bold")
            st.pyplot(fig)

    
            orig_counts = df_filtrado[var].value_counts().sort_index()
            imp_counts = df_filtrado[f"{var}_hd"].value_counts().sort_index()
            orig_counts, imp_counts = orig_counts.align(imp_counts, fill_value=0)
            ct = pd.DataFrame({"original": orig_counts, "imputada": imp_counts}).T

            chi2, p, dof, exp = chi2_contingency(ct)
            v = Cramers_v(ct)

            st.markdown(f"### Resultados estadísticos para {var}")
            st.write("Proporciones antes de imputar:")
            st.write((orig_counts / orig_counts.sum()).round(3))
            st.write("Proporciones después de imputar:")
            st.write((imp_counts / imp_counts.sum()).round(3))
            st.write(f"**Chi² = {chi2:.3f}**, gl = {dof}, **p = {p:.4f}**")
            st.write("Cramér's V =", round(v, 3))
            if p > 0.05:
                st.success("No se rechaza H₀: las distribuciones son estadísticamente iguales.")
            else:
                st.warning("Se rechaza H₀: las distribuciones son diferentes tras imputar.")
            st.markdown("---")
    else:
        st.markdown("no hay datos faltantes en el dataset")
else:
    st.warning("Primero ve a la página de transformación y guarda los datos.")