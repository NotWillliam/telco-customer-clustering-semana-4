# XAI aplicada a la prediccion de Churn en Telecomunicaciones

Repositorio generado a partir del notebook original **ACTIVIDAD4_GRUPO_9.ipynb** y el dataset **Telco Customer Churn**.

## Problema

El objetivo es predecir la cancelacion de clientes (*churn*) en una empresa de telecomunicaciones y explicar las decisiones del modelo usando tecnicas de explicabilidad de inteligencia artificial.

## Metodologia

1. Carga y preprocesamiento del dataset Telco Customer Churn.
2. Entrenamiento de modelos Random Forest y Logistic Regression.
3. Evaluacion con accuracy, recall, F1-score y AUC-ROC.
4. Explicabilidad global con SHAP.
5. Explicabilidad local con SHAP y LIME.
6. Comparacion de tecnicas: SHAP, importancia interna de Random Forest, Permutation Importance y LIME.
7. Analisis interpretativo y reflexion etica.

## Estructura del repositorio

```text
.
├── data/Telco-Customer-Churn.csv
├── notebooks/ACTIVIDAD4_GRUPO_9_XAI_ORIGINAL.ipynb
├── notebooks/ACTIVIDAD4_GRUPO_9_XAI_AMPLIADO_EJECUTADO.ipynb
├── src/xai_visualizations.py
├── results/figures/
├── results/metrics/
├── requirements.txt
└── README.md
```

## Resultados principales

- AUC-ROC Random Forest: **0.8413**
- Accuracy Random Forest: **0.7491**
- Recall Random Forest: **0.7941**
- F1-score Random Forest: **0.6272**

Las variables con mayor influencia se encuentran en `results/metrics/importancia_global_shap.csv`. En general, las variables relacionadas con permanencia, tipo de contrato, cargos y metodo de pago presentan alto impacto en la prediccion.

## Visualizaciones generadas

- `01_variables_mayor_impacto_shap.png`
- `02_comparacion_tecnicas_globales.png`
- `03_comparacion_shap_lime_casos_individuales.png`
- `04_caso_a_decision_explicada.png`
- `05_caso_b_decision_explicada.png`

## Analisis interpretativo

El modelo Random Forest presenta buen rendimiento, pero no es interpretable directamente. SHAP y LIME permiten comprender por que una prediccion se inclina hacia churn o no churn, mejorando la transparencia y la auditabilidad del sistema.

## Riesgos eticos y sociales

Implementar este sistema sin explicabilidad podria ocasionar decisiones comerciales injustas, exclusion de usuarios que requieren atencion o dependencia excesiva del modelo sin revision humana.

## Como ejecutar

```bash
pip install -r requirements.txt
jupyter notebook notebooks/ACTIVIDAD4_GRUPO_9_XAI_AMPLIADO_EJECUTADO.ipynb
```

El dataset tambien se encuentra copiado en la raiz como `Telco-Customer-Churn.csv` para mantener compatibilidad con el codigo original del notebook.
