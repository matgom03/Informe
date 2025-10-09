import streamlit as st


st.title(" Conclusiones generales")


st.markdown("""
            
## Conclusiones generales sobre analisis univariado 
- Se observa la presencia de datos atipicos para las variables numericas y presencia de datos Nan para las variables categoricas que deben ser tratados
- Se observa que algunos de los graficos como el de Age o Fnlwgt tienen un sesgo positivo, por lo que la mayoria de los datos se concentran en los valores inferiores 
- Se observa tambien que hay bastantes valores de 0 en capital gain o capital loss, sin embargo estos valores no los podemos considerar como valores faltantes ya que si son valores reales
- Tambien en las variables categoricas se observa que la variable income esta desbalanceada, esto se debe tener en cuenta si se realiza algun modelo con esta variable como objetivo


---

## Conclusiones generales sobre analisis bivariado 
- Podemos observar que en general, las correlaciones entre las variables numericas del dataset son bastante bajas, tanto en las correlaciones de pearson y spearman ninguno supera mas de 0.2 de correlacion. 
En general al realizar la correlacion con el metodo de spearman se generaron mejores resultados de los que habia a comparación de cuando se calcula con spearman
- Podemos observar que en las variables categoricas, el p-valor es menor que 0.05 por lo que todos los pares tienen relaciones significativas. 
Sin embargo para un mayor entendimiento se calculo tambien la fuerza mediante cramer para ver cuales eran asociadas con mayor fuerza, las cuales en su mayoria son las variables que estaban mas relacionadas en concepto como raza y pais nativo, las mas moderadas tampoco se pueden ignorar porque estas tambien muestran patrones importantes como la relacion entre ingresos y la educacion 
- Se utilizaron las pruebas del test de levene y la normalidad con kstest para determinar si se debia usar ANOVA o Kruskall wallis para el analisis bivariado entre categoricas y numericas, en general al no ser normales y los pares no tener varianzas homogeneas se utilizo kruskall wallis y se nota que en la mayoria de los casos hay diferencias estadisticamente significativas entre los pares, 
lo que significa que hay relaciones fuertes entre ellas, la unica excepcion es  income y Fnlwgt cuyo p-valor es mayor con 0.09 por lo que los ingresos no estan relacionados con el peso de muestra poblacional




---
## Conclusiones generales sobre las correlaciones del dataset
- Podemos observar que en cuanto a las correlaciones de variables hay varias cosas curiosas por ejemplo Education y Education num tienen correlacion perfecta, por lo que si se llega a realizar algun modelado con el dataset se debe eliminar 1 de ellas para no generar data leakage.
- En general las variables categoricas estan muchisimo mas relacionadas que las variables numericas lo cual muestra que las variables numericas son mas independientes entre si en comparacion y se debe tomar en cuenta si se realiza el modelado con este dataset 


---
## Conclusiones generales sobre el tratamiento de NA y outliers
-  se realizo el filtrado a 3 de las 5 variables del dataset porque Age y Fnlwtg son outliers que representan valores reales, en general no hubo un cambio muy grande en las distribuciones de los datos si bien quedan valores extremos estos no seran un problema para los analisis posteriores
-  se realizo la imputacion de datos faltantes, como tienen bajo porcentaje se realizara imputacion por la moda con Native country, para verificar si funciona se utilizara el test de chi cuadrado para la imputacion de valores faltantes, los resultados mostraron que la imputacion de valores faltantes fue exitosa ya que con un p-valor tan alto no se rechaza la Ho y las distribuciones se mantienen
- de la misma manera a las 2 variables restantes se les imputo por Hotdeck y dieron resultados similares siendo imputaciones realizadas de manera exitosa
---
## Conclusiones del dataset
El análisis exploratorio reveló patrones importantes de distribución y asociación, confirmando la necesidad de un tratamiento cuidadoso de valores faltantes y outliers. Las variables categóricas tienen mayor relevancia relacional, mientras que las numéricas son más independientes. Esto servirá como base sólida para futuras etapas de modelado.
""")