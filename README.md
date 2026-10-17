# Predicción de CREDIT_LIMIT (regresión)

**Dataset:** Credit Card Dataset (8,950 clientes, 17 variables; Kaggle `arjunbhasin2013/ccdata`).
**Objetivo:** predecir `CREDIT_LIMIT` a partir del comportamiento de uso de la tarjeta.

## Metodología
EDA → imputación de `MINIMUM_PAYMENTS` (KNN, ajustada solo con train) → eliminación de colinealidad por VIF (`PURCHASES`, `PURCHASES_FREQUENCY`) → escalado según asimetría → Ridge, Lasso, regresión lineal, RandomForest y HistGradientBoosting → GridSearchCV + KFold (5) → análisis de residuos y permutation importance.

## Resultado
Modelo final: HistGradientBoosting (tuned). CV 5-fold: R² 0.531 ± 0.034, RMSE 2,507, MAE 1,734. Test: R² 0.496, RMSE 2,504, MAE 1,746. Empata estadísticamente con RandomForest; los modelos lineales rinden R² ≈ 0.45.

Variables clave: `BALANCE`, `PAYMENTS`, `MINIMUM_PAYMENTS`, `CASH_ADVANCE`.
El modelo sobrepredice límites bajos y subpredice límites altos (> 10,000, MAE ≈ 4,900).

## Limitaciones
Apoyo exploratorio, no apto para fijar límites de forma automática (MAE ≈ 58% de la mediana). Sin línea base ni IC en test; imputación/escalado fuera del pipeline de CV (fuga leve). Moneda no documentada.

## Ejecutar
Colocar `CC GENERAL.csv` junto al notebook (ajustar `path` en la celda 4) y ejecutar `ML_REGRESSION_1.0.6.ipynb`. Requiere pandas, numpy, scikit-learn, statsmodels, seaborn, scipy.
