import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

st.set_page_config(layout='wide', page_title='App Interactiva - Implementacion')

# ---------- Helpers -------------------------------------------------
ADULT_COLUMNS = [
    'age','workclass','fnlwgt','education','education-num','marital-status',
    'occupation','relationship','race','sex','capital-gain','capital-loss',
    'hours-per-week','native-country','income'
]

def load_data():
    try:
        df1 = pd.read_csv('adult.data', header=None, na_values=['?', ' ?'], skipinitialspace=True)
        df2 = pd.read_csv('adult.test', header=None, na_values=['?', ' ?'], skipinitialspace=True, skiprows=1)
        df = pd.concat([df1, df2], ignore_index=True)
        if df.shape[1] == len(ADULT_COLUMNS):
            df.columns = ADULT_COLUMNS
        return df
    except FileNotFoundError:
        st.error('No se encontraron los archivos adult.data / adult.test en el directorio de trabajo.')
        return None

def cramers_v(confusion_matrix):
    chi2, p, dof, expected = chi2_contingency(confusion_matrix)
    n = confusion_matrix.values.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    try:
        return np.sqrt(phi2 / min(k-1, r-1))
    except Exception:
        return np.nan

def impute_mode(df, cols):
    df2 = df.copy()
    modes = {}
    for c in cols:
        mode = df2[c].mode(dropna=True)
        if len(mode)>0:
            modes[c] = mode.iloc[0]
            df2[c] = df2[c].fillna(modes[c])
        else:
            modes[c] = np.nan
    return df2, modes

def hotdeck_impute(df, target_col, ref_col=None):
    df2 = df.copy()
    global_mode = df2[target_col].mode(dropna=True)
    global_mode = global_mode.iloc[0] if len(global_mode)>0 else np.nan
    if ref_col is None:
        df2[target_col] = df2[target_col].fillna(global_mode)
        return df2
    mapping = {}
    for val, group in df2.groupby(ref_col):
        m = group[target_col].mode(dropna=True)
        mapping[val] = (m.iloc[0] if len(m)>0 else np.nan)
    na_idx = df2[df2[target_col].isna()].index
    for i in na_idx:
        ref_val = df2.loc[i, ref_col]
        if pd.isna(ref_val) or mapping.get(ref_val, np.nan) is np.nan:
            df2.at[i, target_col] = global_mode
        else:
            df2.at[i, target_col] = mapping[ref_val]
    return df2

# ---------- App UI -------------------------------------------------
st.title('App interactiva basada en Implementacion.ipynb')
st.markdown('Esta app usa los mismos datos que el notebook original (adult.data + adult.test).')

with st.sidebar:
    st.header('Controles')
    action = st.radio('Selecciona acción:', ['Contexto del EDA','Explorar datos','Imputar valores','Bivariante categórica','VIF y correlaciones','Descargar df procesado'])

# cargar datos
df = load_data()
if df is None:
    st.stop()

if action == 'Contexto del EDA':
    st.header('Contexto del EDA')
    st.markdown('''
    En el presente trabajo se realiza el análisis del siguiente dataset de la página de UCI Machine Learning Repository. El dataset es conocido como el **Adult dataset** o también como **Census Income**. Fue publicado en el UCI Machine Learning Repository por Barry Becker en 1996, a partir de datos del censo de EE. UU. de 1994. Tiene **48 842 instancias y 14 atributos**.

    **Fuente de los datos:** [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)

    ### Problema
    El dataset busca predecir si el ingreso anual de una persona es **mayor o menor a 50.000 USD**, basándose en características demográficas, educativas y laborales.

    ### Objetivos
    - Limpiar y transformar el dataset original (manejo de nulos y categorización).
    - Explorar la distribución de las variables numéricas y categóricas.
    - Identificar relaciones entre nivel educativo, ocupación, horas de trabajo y nivel de ingresos.
    - Preparar un dataset listo para modelado predictivo.

    ### Variables incluidas
    - Age
    - Fnlwgt (final weight)
    - Education
    - Education-num
    - Capital-gain
    - Capital-loss
    - Hours-per-week
    - Workclass
    - Occupation
    - Native-country
    - Relationship
    - Marital-status
    - Race
    - Sex
    - Income

    ### Diccionario breve
    - **fnlwgt:** Peso de muestra utilizado por la Oficina del Censo para representar la población.
    - **Education-num:** Codifica la educación como número de años de estudio.
    - **Capital-gain / Capital-loss:** Medidas financieras adicionales.
    - **Income:** Variable objetivo binaria.

    ### Alcance
    El análisis se limita a describir los patrones y relaciones en los datos censales.
    No se incluyen modelos predictivos complejos, aunque el dataset puede servir para modelos de clasificación en futuras entregas.
    ''')

elif action == 'Explorar datos':
    st.subheader('Vista rápida')
    st.dataframe(df.head(100))
    st.write('Dimensiones:', df.shape)
    st.subheader('Resumen de valores faltantes')
    miss = df.isna().sum().sort_values(ascending=False)
    st.dataframe(miss[miss>0])
    st.subheader('Distribuciones (selecciona columna)')
    col = st.selectbox('Columna para visualizar', df.columns.tolist())
    if df[col].dtype == 'O' or df[col].nunique() < 30:
        counts = df[col].value_counts(dropna=False)
        fig, ax = plt.subplots()
        counts.plot.bar(ax=ax)
        ax.set_title(col)
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        df[col].hist(ax=ax, bins=30)
        ax.set_title(col)
        st.pyplot(fig)

elif action == 'Imputar valores':
    st.subheader('Imputación interactiva')
    cols = st.multiselect('Columnas a imputar (categoricas sugeridas)', df.select_dtypes(include=['object','category']).columns.tolist())
    method = st.selectbox('Método', ['Moda (global)','Hot-deck (usar 1 columna de referencia)'])
    if st.button('Ejecutar imputación'):
        if method == 'Moda (global)':
            df_imp, modes = impute_mode(df, cols)
            st.success('Imputación por moda aplicada')
            st.write('Modas usadas:')
            st.json(modes)
            st.subheader('Comparación antes/después (tasas)')
            for c in cols:
                before = df[c].value_counts(normalize=True, dropna=False)
                after = df_imp[c].value_counts(normalize=True, dropna=False)
                comp = pd.concat([before, after], axis=1).fillna(0)
                comp.columns = ['antes','despues']
                st.write('---')
                st.write(c)
                st.dataframe(comp)
            st.session_state['df'] = df_imp
        else:
            ref = st.selectbox('Columna de referencia para hotdeck', [None]+df.select_dtypes(include=['object','category']).columns.tolist())
            if st.button('Aplicar hotdeck'):
                df_imp = df.copy()
                for c in cols:
                    df_imp = hotdeck_impute(df_imp, c, ref)
                st.success('Hot-deck aplicado')
                st.session_state['df'] = df_imp
                st.write('Comparación tasas (antes/despues):')
                for c in cols:
                    before = df[c].value_counts(normalize=True, dropna=False)
                    after = df_imp[c].value_counts(normalize=True, dropna=False)
                    comp = pd.concat([before, after], axis=1).fillna(0)
                    comp.columns = ['antes','despues']
                    st.write('---')
                    st.write(c)
                    st.dataframe(comp)

elif action == 'Bivariante categórica':
    st.subheader('Chi-cuadrado y Cramér')
    col_x = st.selectbox('Variable A (categorica)', df.select_dtypes(include=['object','category']).columns.tolist(), index=0)
    col_y = st.selectbox('Variable B (categorica)', df.select_dtypes(include=['object','category']).columns.tolist(), index=1)
    if st.button('Calcular prueba'):
        ct = pd.crosstab(df[col_x], df[col_y])
        chi2, p, dof, exp = chi2_contingency(ct)
        v = cramers_v(ct)
        st.write('Chi2 =', round(chi2,4), ' p =', round(p,6), ' dof=', dof)
        st.write("Cramér's V ≈", round(v,4))
        st.write('Tabla de contingencia (extracto):')
        st.dataframe(ct)

elif action == 'VIF y correlaciones':
    st.subheader('Correlaciones y VIF (numéricas)')
    num = df.select_dtypes(include=['int64','float64']).copy()
    if num.shape[1] == 0:
        st.write('No hay variables numéricas en el dataset.')
    else:
        st.write('Matriz de correlación (pearson):')
        corr = num.corr()
        st.dataframe(corr)
        st.write('Matriz de correlación (spearman):')
        st.dataframe(num.corr(method='spearman'))
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            X = num.dropna()
            vif_data = pd.DataFrame({'variable': X.columns, 'VIF':[variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
            vif_data = vif_data.sort_values('VIF', ascending=False)
            st.write('VIF (orden descendente):')
            st.dataframe(vif_data)
        except Exception as e:
            st.error('Error calculando VIF: '+str(e))

elif action == 'Descargar df procesado':
    st.subheader('Descargar')
    df_out = st.session_state.get('df', df)
    st.write('Dimensiones:', df_out.shape)
    csv = df_out.to_csv(index=False).encode('utf-8')
    st.download_button('Descargar CSV', csv, 'adult_processed.csv', 'text/csv')

st.markdown('---')
st.caption('App generada automáticamente desde Implementacion.ipynb — versión interactiva con contexto del dataset.')

