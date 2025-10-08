import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro, normaltest, kstest,f_oneway, kruskal, chi2_contingency, levene,mannwhitneyu
from itertools import combinations
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st
@st.cache_data
def load_data():
    """Carga y concatena los archivos adult.data y adult.test."""
    df1=pd.read_csv("adult.data",header=None,na_values=["?", " ?"],skipinitialspace=True)
    df2=pd.read_csv("adult.test",header=None,na_values=["?", " ?"],skipinitialspace=True,skiprows=1)
    df = pd.concat([df1, df2], ignore_index=True)
    cols = ["Age",
          "Workclass",
          "Fnlwgt",
          "Education",
          "Education-num",
          "Marital-status",
          "Occupation",
          "Relationship",
          "Race",
          "Sex",
          "Capital-gain",
          "Capital-loss",
          "Hours-per-week",
          "Native-country",
          "Income"]
    df.columns = cols
    return df1.head(), df2.head(), df
def Summary(data, sheet):
    st.subheader(f"Hoja: {sheet}")

    # Crear la tabla de resumen
    resumen = {
        "Cantidad de filas": data.shape[0],
        "Cantidad de columnas": data.shape[1],
        "Datos faltantes": data.isnull().sum().sum(),
        "Filas duplicadas": data.duplicated().sum()
    }

    # Convertir el resumen en un DataFrame
    ResumenHoja = pd.DataFrame(resumen, index=["Resumen"])

    st.markdown("### Resumen del dataset")
    st.dataframe(ResumenHoja)

    st.markdown("### Primeras filas del dataset")
    st.dataframe(data.head())
def analizar_categoricas(df):
    resultados = []
    categoricas = df.select_dtypes(include=['object', 'category', 'bool']).columns

    for col in categoricas:
        conteo = df[col].value_counts(dropna=False)
        proporcion = df[col].value_counts(normalize=True, dropna=False)
        nulos = df[col].isna().sum()

        categoria_moda = conteo.index[0] if not conteo.empty else None
        frecuencia_moda = conteo.iloc[0] if not conteo.empty else None

        resultados.append({
            "variable": col,
            "cantidad_categorias": df[col].nunique(dropna=False),
            "categoria_mas_frecuente": categoria_moda,
            "frecuencia_mas_frecuente": frecuencia_moda,
            "porcentaje_mas_frecuente": proporcion.iloc[0] if not conteo.empty else None,
            "nulos": nulos
        })

    return pd.DataFrame(resultados).sort_values(by="cantidad_categorias", ascending=False)
def GraficarCategoricas(df, tipo="barra"):
    categoricas = df.select_dtypes(include=['object', 'category', 'bool']).columns
    if len(categoricas) == 0:
        st.warning("No se detectaron variables categóricas en el DataFrame.")
        return

    for col in categoricas:
        conteo = df[col].value_counts(dropna=False)

        if tipo == "barra":
            fig, ax = plt.subplots(figsize=(8, 6))
            conteo.plot(kind="bar", color="skyblue", edgecolor="black", ax=ax)
            ax.set_title(f"Frecuencia de {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frecuencia")
            st.pyplot(fig)

        elif tipo == "pastel":
            fig, ax = plt.subplots(figsize=(6, 6))
            conteo.plot(kind="pie", autopct='%1.1f%%', colors=plt.cm.Set3.colors, startangle=90, ax=ax)
            ax.set_ylabel("")
            ax.set_title(f"Distribución de {col}")
            st.pyplot(fig)
        else:
            st.error("El parámetro 'tipo' debe ser 'barra' o 'pastel'.")
def resumen_numericas(df):
    numericas = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numericas) == 0:
        st.warning("No se detectaron variables numéricas en el DataFrame.")
        return None

    resumen = df[numericas].describe().T
    resumen["varianza"] = df[numericas].var()
    resumen["mediana"] = df[numericas].median()
    return resumen
def graficar_numericas(df):
    numericas = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numericas) == 0:
        st.warning("No se detectaron variables numéricas en el DataFrame.")
        return

    for col in numericas:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        df[col].hist(bins=20, color="skyblue", edgecolor="black", ax=axes[0])
        axes[0].set_title(f"Histograma de {col}")
        axes[1].boxplot(df[col].dropna(), vert=False, patch_artist=True,
                        boxprops=dict(facecolor="lightgreen", color="black"),
                        medianprops=dict(color="red"))
        axes[1].set_title(f"Boxplot de {col}")
        st.pyplot(fig)
def detectar_outliers(df):
    resultados = []
    numericas = df.select_dtypes(include=[np.number]).columns

    for col in numericas:
        serie = df[col].dropna()
        if len(serie) == 0:
            continue

        q1, q3 = serie.quantile([0.25, 0.75])
        iqr = q3 - q1
        lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((serie < lim_inf) | (serie > lim_sup)).sum()
        total = len(serie)

        resultados.append({
            "variable": col,
            "n_muestras": total,
            "outliers": outliers,
            "proporcion_outliers": outliers / total if total > 0 else np.nan
        })

    return pd.DataFrame(resultados).sort_values(by="proporcion_outliers", ascending=False)
def analisis_bivariado_numerico(df, top=5, mostrar_graficos=True):
    # Seleccionar solo las columnas numéricas
    num_df = df.select_dtypes(include=[np.number])
    cols = num_df.columns

    if len(cols) < 2:
        st.error("Se necesitan al menos dos variables numéricas para el análisis bivariado.")
        return None

    resultados = []

    # Calcular correlaciones entre todas las parejas posibles
    for var1, var2 in combinations(cols, 2):
        pearson = num_df[var1].corr(num_df[var2], method='pearson')
        spearman = num_df[var1].corr(num_df[var2], method='spearman')
        covarianza = np.cov(num_df[var1].dropna(), num_df[var2].dropna())[0, 1]

        resultados.append({
            'Variable 1': var1,
            'Variable 2': var2,
            'Correlación Pearson': pearson,
            'Correlación Spearman': spearman,
            'Covarianza': covarianza
        })

    corr_df = pd.DataFrame(resultados).sort_values(by='Correlación Spearman', ascending=False)

    # Mostrar resultados en Streamlit
    st.subheader(" Correlaciones más altas")
    st.dataframe(corr_df.head(top))

    st.subheader(" Correlaciones más bajas")
    st.dataframe(corr_df.tail(top))

    # Mostrar gráficos si se solicita
    if mostrar_graficos:
        st.subheader(" Gráficos de correlaciones más altas y más bajas")
        pares_altas = corr_df.head(top)
        pares_bajas = corr_df.tail(top)
        pares_interes = pd.concat([pares_altas, pares_bajas])

        for _, row in pares_interes.iterrows():
            var1, var2 = row['Variable 1'], row['Variable 2']
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.regplot(x=var1, y=var2, data=num_df, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'}, ax=ax)
            ax.set_title(f"{var1} vs {var2}\nCorrelación Spearman: {row['Correlación Spearman']:.3f}")
            st.pyplot(fig)

    return corr_df
def cramers_v(x, y):
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
def Cramers_v(ct):
    chi2, _, _, _ = chi2_contingency(ct, correction=False)
    n = ct.to_numpy().sum()
    k = min(ct.shape)-1
    return np.sqrt(chi2 / (n * k)) if n > 0 and k > 0 else np.nan
def analisis_bivariado_categorico(df, top=5, mostrar_graficos=True, grafico='heatmap'):
    # Filtrar solo variables categóricas
    cat_df = df.select_dtypes(include=['object', 'category'])
    cols = cat_df.columns.tolist()

    if len(cols) < 2:
        st.error("Se necesitan al menos dos variables categóricas para el análisis bivariado.")
        return None

    resultados = []

    # Iterar sobre todas las combinaciones posibles de variables categóricas
    for var1, var2 in combinations(cols, 2):
        # Crear tabla de contingencia
        tabla = pd.crosstab(df[var1], df[var2])

        # Evitar tablas muy pequeñas
        if tabla.shape[0] < 2 or tabla.shape[1] < 2:
            continue

        # Prueba Chi-cuadrado
        chi2, p, dof, expected = chi2_contingency(tabla)

        # Calcular V de Cramer
        n = tabla.sum().sum()
        phi2 = chi2 / n
        r, k = tabla.shape
        phi2corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1))
        rcorr = r - ((r - 1)**2) / (n - 1)
        kcorr = k - ((k - 1)**2) / (n - 1)
        cramer_v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))) if min(kcorr, rcorr) > 1 else 0

        resultados.append({
            'Variable 1': var1,
            'Variable 2': var2,
            'Chi-cuadrado': chi2,
            'Grados de libertad': dof,
            'Valor p': p,
            'V de Cramer': cramer_v
        })

    # Convertir resultados a DataFrame
    resumen_df = pd.DataFrame(resultados).sort_values(by='Valor p', ascending=True).reset_index(drop=True)

    if resumen_df.empty:
        st.warning("No se encontraron combinaciones categóricas válidas para el análisis.")
        return resumen_df

    # Mostrar top relaciones significativas y menos significativas
    st.subheader(" Relaciones categóricas más significativas (menor valor p)")
    st.dataframe(resumen_df.head(top))

    st.subheader(" Relaciones categóricas menos significativas")
    st.dataframe(resumen_df.tail(top))

    # Graficar top relaciones más significativas
    if mostrar_graficos:
        st.subheader(" Gráficos de las relaciones categóricas más significativas")

        top_pairs = resumen_df.head(top)

        for _, row in top_pairs.iterrows():
            var1, var2 = row['Variable 1'], row['Variable 2']
            st.markdown(f"**{var1} vs {var2}** (p = {row['Valor p']:.4f}, V de Cramer = {row['V de Cramer']:.3f})")

            # Normalizar por filas para mostrar proporciones
            tabla_rel = pd.crosstab(df[var1], df[var2], normalize='index')

            fig, ax = plt.subplots(figsize=(7, 5))
            if grafico == 'heatmap':
                sns.heatmap(tabla_rel, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax)
                ax.set_title(f"{var1} vs {var2}\n(p = {row['Valor p']:.4f}, V = {row['V de Cramer']:.3f})")
                ax.set_xlabel(var2)
                ax.set_ylabel(var1)
                plt.tight_layout()
                st.pyplot(fig)

    return resumen_df
def analisis_bivariado_cat_num(df, categoricas=None, numericas=None, mostrar_graficos=True):
    if categoricas is None:
        categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if numericas is None:
        numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if len(categoricas) == 0 or len(numericas) == 0:
        st.error("Debe haber al menos una variable categórica y una numérica.")
        return None

    resultados = []

    for cat in categoricas:
        for num in numericas:
            grupos = [df[num][df[cat] == nivel].dropna() for nivel in df[cat].dropna().unique()]
            grupos = [g for g in grupos if len(g) > 1]
            if len(grupos) < 2:
                continue

            # ==== Normalidad (Kolmogorov-Smirnov) ====
            normalidades = []
            for g in grupos:
                g_std = (g - np.mean(g)) / np.std(g, ddof=1)
                _, p_norm = kstest(g_std, 'norm')
                normalidades.append(p_norm > 0.05)

            # ==== Homocedasticidad (Levene) ====
            stat_lev, p_levene = levene(*grupos)
            homocedasticas = p_levene > 0.05

            # ==== Elección de prueba ====
            if all(normalidades) and homocedasticas:
                prueba = "ANOVA"
                stat, p_valor = f_oneway(*grupos)
            else:
                prueba = "Kruskal-Wallis"
                stat, p_valor = kruskal(*grupos)

            resultados.append({
                'Categórica': cat,
                'Numérica': num,
                'Prueba': prueba,
                'Normalidad_OK': all(normalidades),
                'Homocedasticidad_OK': homocedasticas,
                'p-Levene': p_levene,
                'Estadístico': stat,
                'p-valor': p_valor
            })

            # ==== Gráfico ====
            if mostrar_graficos:
                st.markdown(f"**{num} por {cat}** ({prueba}, p = {p_valor:.4f})")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(x=cat, y=num, hue=cat, data=df, palette="Set2", legend=False, ax=ax)
                ax.set_title(f'{num} por {cat}\n({prueba}, p = {p_valor:.4f})')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)

    if not resultados:
        st.warning("No se encontraron combinaciones válidas con suficientes datos.")
        return None

    resultados_df = pd.DataFrame(resultados).sort_values(by='p-valor')
    st.subheader(" Resultados del análisis bivariado categórica vs numérica")
    st.dataframe(resultados_df)

    return resultados_df
def prueba_normalidad(df, alpha=0.05):
    resultados = []

    # Detectar variables numéricas
    numericas = df.select_dtypes(include=[np.number]).columns

    for col in numericas:
        serie = df[col].dropna()

        if len(serie) < 8:
            resultados.append({
                "variable": col,
                "n_muestras": len(serie),
                "shapiro_p": np.nan,
                "dagostino_p": np.nan,
                "ks_p": np.nan,
                f"es_normal (alpha={alpha})": "Muestra insuficiente"
            })
            continue

        # Shapiro-Wilk
        stat_shapiro, p_shapiro = shapiro(serie) if len(serie) <= 5000 else (np.nan, np.nan)

        # D’Agostino y Pearson
        stat_dagostino, p_dagostino = normaltest(serie)

        # Kolmogorov-Smirnov
        stat_ks, p_ks = kstest(
            (serie - serie.mean()) / serie.std(ddof=0), 'norm'
        )

        # Veredicto
        es_normal = (
            (np.isnan(p_shapiro) or p_shapiro > alpha) and
            (p_dagostino > alpha) and
            (p_ks > alpha)
        )

        resultados.append({
            "variable": col,
            "n_muestras": len(serie),
            "shapiro_p": p_shapiro,
            "dagostino_p": p_dagostino,
            "ks_p": p_ks,
            f"es_normal (alpha={alpha})": es_normal
        })

    return pd.DataFrame(resultados).sort_values(by=f"es_normal (alpha={alpha})", ascending=False)
def analizar_colinealidad_y_correlaciones(df, umbral_vif=10, mostrar_graficos=True):
    resultados = {}

    # Variables numéricas y categóricas
    num_vars = df.select_dtypes(include=['int64', 'float64']).columns
    cat_vars = df.select_dtypes(include=['object', 'category']).columns

    # ===============================
    # CORRELACIONES NUMÉRICAS
    # ===============================
    if len(num_vars) > 1:
        corr_pearson = df[num_vars].corr(method='pearson')
        corr_spearman = df[num_vars].corr(method='spearman')
        resultados['corr_pearson'] = corr_pearson
        resultados['corr_spearman'] = corr_spearman

        if mostrar_graficos:
            st.subheader("Correlación numérica (Pearson)")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr_pearson, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax1)
            st.pyplot(fig1)

            st.subheader("Correlación numérica (Spearman)")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr_spearman, annot=True, fmt=".2f", cmap='vlag', center=0, ax=ax2)
            st.pyplot(fig2)

        st.write("Matriz de correlación (Pearson):")
        st.dataframe(corr_pearson)

        st.write("Matriz de correlación (Spearman):")
        st.dataframe(corr_spearman)

    # ===============================
    # COLINEALIDAD (VIF)
    # ===============================
    if len(num_vars) > 1:
        X = df[num_vars].dropna()
        vif_data = pd.DataFrame({
            "Variable": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        })
        vif_data["Colinealidad"] = np.where(vif_data["VIF"] > umbral_vif, "Alta", "Aceptable")
        resultados['vif'] = vif_data

        st.subheader("Factor de inflación de varianza (VIF)")
        st.write(f"Umbral VIF: {umbral_vif}")
        st.dataframe(vif_data)

    # ===============================
    # CORRELACIONES CATEGÓRICAS (Cramér’s V)
    # ===============================
    if len(cat_vars) > 1:
        matriz_cramer = pd.DataFrame(
            np.ones((len(cat_vars), len(cat_vars))),
            index=cat_vars, columns=cat_vars
        )
        for var1, var2 in combinations(cat_vars, 2):
            val = cramers_v(df[var1], df[var2])
            matriz_cramer.loc[var1, var2] = val
            matriz_cramer.loc[var2, var1] = val
        resultados['cramers_v'] = matriz_cramer

        if mostrar_graficos:
            st.subheader("Correlación entre variables categóricas (Cramérs V)")
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            sns.heatmap(matriz_cramer, annot=True, fmt=".2f", cmap='YlGnBu', center=0, ax=ax3)
            st.pyplot(fig3)

        st.write("Matriz de correlación Cramérs V:")
        st.dataframe(matriz_cramer)

    return resultados
def winsorizar_outliers(df, variables=None, lower_percentile=0.01, upper_percentile=0.99, mostrar_resumen=True):
    if variables is None:
        variables = ['Capital-gain', 'Capital-loss', 'Hours-per-week']

    df_wins = df.copy()
    resumen = {}

    for var in variables:
        if var not in df.columns:
            continue

        lower = df[var].quantile(lower_percentile)
        upper = df[var].quantile(upper_percentile)
        df_wins[var] = np.clip(df[var], lower, upper)
        resumen[var] = {
            'min_original': df[var].min(),
            'max_original': df[var].max(),
            'lower_limit': lower,
            'upper_limit': upper,
            'min_winsorizado': df_wins[var].min(),
            'max_winsorizado': df_wins[var].max()
        }

    if mostrar_resumen:
        st.subheader("Resumen de Winsorización")
        st.dataframe(pd.DataFrame(resumen).T)

    return df_wins
def hot_deck_group(df, col, group, random_state=42):
    rng = np.random.default_rng(random_state)
    out = df[col].copy()
    for g, sub in df.groupby(group):
        pool = sub[col].dropna().to_numpy()
        idx = sub.index[sub[col].isna()]
        if pool.size > 0 and len(idx) > 0:
            out.loc[idx] = rng.choice(pool, size=len(idx), replace=True)
    return out