import streamlit as st


st.title(" Contexto del EDA - Adult Dataset")


st.markdown("""
## Contexto del EDA


En el presente trabajo se realiza el análisis del siguiente dataset de la página de **UCI Machine Learning Repository**,
conocido como **Adult dataset** o **Census Income**.
Fue publicado por Barry Becker en 1996, a partir de datos del censo de EE. UU. de 1994.
Contiene **48,842 instancias** y **14 atributos**.


 **Fuente de los datos:** [https://archive.ics.uci.edu/dataset/2/adult](https://archive.ics.uci.edu/dataset/2/adult)


---


##  Problema
El dataset busca predecir si el **ingreso anual de una persona es mayor o menor a 50,000 USD**,
basándose en características demográficas, educativas y laborales.


---


##  Objetivos
- Limpiar y transformar el dataset original (manejo de nulos y categorización).
- Explorar la distribución de las variables numéricas y categóricas.
- Identificar relaciones entre nivel educativo, ocupación, horas de trabajo y nivel de ingresos.
- Preparar un dataset listo para modelado predictivo.


---


##  Variables incluidas
- **Age**
- **Fnlwgt** (final weight)
- **Education**
- **Education-num**
- **Capital-gain**
- **Capital-loss**
- **Hours-per-week**
- **Workclass**
- **Occupation**
- **Native-country**
- **Relationship**
- **Marital-status**
- **Race**
- **Sex**
- **Income**


---


##  Diccionario breve
- **fnlwgt:** Peso de muestra utilizado por la Oficina del Censo para representar la población.
- **Education-num:** Codifica la educación como número de años de estudio.
- **Capital-gain / Capital-loss:** Medidas financieras adicionales.
- **Income:** Variable objetivo binaria.


---


##  Alcance
El análisis se limita a describir los patrones y relaciones en los datos censales.
No se incluyen modelos predictivos complejos, aunque el dataset puede servir
para **modelos de clasificación** en una entrega posterior.
""")