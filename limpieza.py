#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Limpieza de datos para predicción del APGAR a los 5 minutos.
Este script realiza la limpieza y preparación de datos del dataset de nacimientos
del Hospital General de Medellín.
"""

# Importar bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Configuración
warnings.filterwarnings('ignore')
plt.style.use('seaborn')
pd.set_option('display.max_columns', None)

# ==============================================================================
# 3.1 INTEGRACIÓN
# ==============================================================================
print("="*80)
print("3.1 INTEGRACIÓN")
print("="*80)

# Cargar el dataset
file_path = 'Nacimientos_ocurridos_en_el_Hospital_General_de_Medell_n_20250517.csv'
try:
    df = pd.read_csv(file_path, encoding='latin1')
    print(f"Dataset cargado exitosamente. Dimensiones: {df.shape}")
    
    # Mostrar las primeras filas
    print("\nPrimeras 5 filas del dataset:")
    print(df.head())
except Exception as e:
    print(f"Error al cargar el dataset: {e}")
    exit()

# ==============================================================================
# 3.2 SELECCIÓN DE VARIABLES
# ==============================================================================
print("\n" + "="*80)
print("3.2 SELECCIÓN DE VARIABLES")
print("="*80)

# Mostrar todas las columnas disponibles
print("\nColumnas disponibles en el dataset:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

# Variables relevantes para la predicción del APGAR2
relevant_columns = [
    'SEXO', 'PESO (Gramos)', 'TALLA (CentImetros)', 'TIEMPO DE GESTACION',
    'NUMERO CONSULTAS PRENATALES', 'TIPO PARTO', 'MULTIPLICIDAD EMBARAZO',
    'APGAR1', 'APGAR2', 'EDAD MADRE', 'NIVEL EDUCATIVO MADRE',
    'NUMERO HIJOS NACIDOS VIVOS', 'NUMERO EMBARAZOS', 'REGIMEN SEGURIDAD'
]

# Crear un subconjunto del dataframe con las variables seleccionadas
df_selected = df[relevant_columns].copy()
print(f"\nVariables seleccionadas: {len(relevant_columns)}")
print(f"Dimensiones del dataset reducido: {df_selected.shape}")

# ==============================================================================
# 3.3 DESCRIPCIÓN ESTADÍSTICA
# ==============================================================================
print("\n" + "="*80)
print("3.3 DESCRIPCIÓN ESTADÍSTICA")
print("="*80)

# Obtener descripción estadística para variables numéricas
print("\nEstadísticas descriptivas para variables numéricas:")
desc_stats = df_selected.describe().T
print(desc_stats)

# Análisis de variables categóricas
categorical_cols = df_selected.select_dtypes(include=['object']).columns
print("\nVariables categóricas encontradas:")
print(categorical_cols.tolist())

for col in categorical_cols:
    print(f"\nDistribución de {col}:")
    value_counts = df_selected[col].value_counts(dropna=False)
    print(value_counts.head(10))  # Mostrar solo las 10 categorías más frecuentes
    print(f"Número de categorías únicas: {df_selected[col].nunique()}")

# Visualización de la distribución del APGAR score a los 5 minutos
plt.figure(figsize=(10, 6))
sns.histplot(df_selected['APGAR2'].dropna(), kde=True, bins=20)
plt.title('Distribución del APGAR Score a los 5 minutos')
plt.axvline(x=7, color='r', linestyle='--', label='Score 7 (punto de corte clínico)')
plt.legend()
plt.savefig('apgar2_distribucion.png')
print("\nGráfico de distribución del APGAR2 guardado como 'apgar2_distribucion.png'")

# Verificar si sería mejor un enfoque de clasificación o regresión
unique_values = df_selected['APGAR2'].unique()
print(f"\nValores únicos del APGAR a los 5 minutos: {sorted(unique_values)}")
print(f"Número total de valores únicos: {len(unique_values)}")
if len(unique_values) <= 10:
    print("Recomendación: Enfoque de CLASIFICACIÓN (pocos valores discretos)")
else:
    print("Recomendación: Evaluar enfoque de REGRESIÓN (muchos valores continuos)")

# ==============================================================================
# 3.4 LIMPIEZA DE ATÍPICOS
# ==============================================================================
print("\n" + "="*80)
print("3.4 LIMPIEZA DE ATÍPICOS")
print("="*80)

# Identificación de valores atípicos en variables numéricas
numeric_cols = df_selected.select_dtypes(include=[np.number]).columns
print(f"\nVariables numéricas para análisis de atípicos: {numeric_cols.tolist()}")

# Crear boxplots para visualizar outliers
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df_selected[col])
    plt.title(f'Boxplot de {col}')
plt.tight_layout()
plt.savefig('boxplots_variables_numericas.png')
print("\nBoxplots de variables numéricas guardados como 'boxplots_variables_numericas.png'")

# Función para detectar outliers usando el método IQR
def detect_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    return outliers, lower_bound, upper_bound

# Verificamos outliers en columnas numéricas relevantes
for col in ['PESO (Gramos)', 'TALLA (CentImetros)', 'TIEMPO DE GESTACION', 'EDAD MADRE']:
    outliers, lb, ub = detect_outliers_iqr(df_selected, col)
    print(f"\nOutliers en {col}:")
    print(f"Límite inferior: {lb:.2f}, Límite superior: {ub:.2f}")
    print(f"Número de outliers: {len(outliers)}")
    if len(outliers) > 0 and len(outliers) < 10:
        print("Ejemplos de outliers:")
        print(outliers.head())

# Tratar outliers: Eliminar outliers extremos que no tienen sentido clínico
# Definir límites clínicamente aceptables
clinical_limits = {
    'PESO (Gramos)': (500, 6000),  # Entre 500g y 6kg
    'TALLA (CentImetros)': (30, 60),  # Entre 30cm y 60cm
    'TIEMPO DE GESTACION': (22, 45),  # Entre 22 y 45 semanas
    'EDAD MADRE': (10, 60)  # Entre 10 y 60 años
}

# Filtrar valores fuera de límites clínicos
print("\nEliminación de outliers extremos basado en límites clínicos:")
rows_before = df_selected.shape[0]
for col, (min_val, max_val) in clinical_limits.items():
    mask = (df_selected[col] >= min_val) & (df_selected[col] <= max_val)
    excluded = df_selected.shape[0] - mask.sum()
    print(f"Eliminando {excluded} registros con {col} fuera del rango ({min_val}, {max_val})")
    df_selected = df_selected[mask]

rows_after = df_selected.shape[0]
print(f"\nTamaño del dataframe después de eliminar outliers extremos: {df_selected.shape}")
print(f"Se eliminaron {rows_before - rows_after} registros en total.")

# ==============================================================================
# 3.5 LIMPIEZA DE NULOS
# ==============================================================================
print("\n" + "="*80)
print("3.5 LIMPIEZA DE NULOS")
print("="*80)

# Verificar valores nulos
null_counts = df_selected.isnull().sum()
null_percentages = 100 * df_selected.isnull().mean()

# Crear un dataframe con la información de nulos
null_info = pd.DataFrame({
    'Nulos': null_counts,
    'Porcentaje (%)': null_percentages
})

print("\nAnálisis de valores nulos:")
print(null_info[null_info['Nulos'] > 0].sort_values('Nulos', ascending=False))

# Visualizar el mapa de calor de valores nulos
plt.figure(figsize=(12, 8))
sns.heatmap(df_selected.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Mapa de calor de valores nulos')
plt.savefig('mapa_calor_nulos.png')
print("\nMapa de calor de valores nulos guardado como 'mapa_calor_nulos.png'")

# Estrategia de tratamiento de nulos
print("\nEstrategie de tratamiento de valores nulos:")

# 1. Para variables categóricas: rellenar con la moda
for col in categorical_cols:
    if df_selected[col].isnull().sum() > 0:
        mode_value = df_selected[col].mode()[0]
        df_selected[col].fillna(mode_value, inplace=True)
        print(f"- Valores nulos en {col} reemplazados con la moda: '{mode_value}'")

# 2. Para variables numéricas: rellenar con la mediana
for col in numeric_cols:
    if df_selected[col].isnull().sum() > 0:
        median_value = df_selected[col].median()
        df_selected[col].fillna(median_value, inplace=True)
        print(f"- Valores nulos en {col} reemplazados con la mediana: {median_value}")

# Verificar que no queden nulos
total_nulls = df_selected.isnull().sum().sum()
print(f"\nTotal de valores nulos después del tratamiento: {total_nulls}")

# ==============================================================================
# 3.6 CREACIÓN DE NUEVAS VARIABLES
# ==============================================================================
print("\n" + "="*80)
print("3.6 CREACIÓN DE NUEVAS VARIABLES")
print("="*80)

# 1. Clasificar el peso al nacer en categorías
df_selected['CATEGORIA_PESO'] = pd.cut(
    df_selected['PESO (Gramos)'], 
    bins=[0, 1500, 2500, 4000, 10000],
    labels=['Muy bajo peso', 'Bajo peso', 'Normal', 'Macrosomía']
)

# 2. Clasificar el tiempo de gestación
df_selected['CATEGORIA_GESTACION'] = pd.cut(
    df_selected['TIEMPO DE GESTACION'],
    bins=[0, 28, 37, 42, 50],
    labels=['Muy prematuro', 'Prematuro', 'A término', 'Postérmino']
)

# 3. Crear variable para clasificar APGAR1 (a 1 minuto) en categorías de riesgo
df_selected['RIESGO_APGAR1'] = pd.cut(
    df_selected['APGAR1'],
    bins=[-1, 3, 6, 10],  # -1 para incluir el 0
    labels=['Alto riesgo', 'Riesgo moderado', 'Normal']
)

# 4. Crear variable objetivo categorizada (para clasificación)
df_selected['APGAR2_CAT'] = pd.cut(
    df_selected['APGAR2'],
    bins=[-1, 3, 6, 10],  # -1 para incluir el 0
    labels=['Depresión severa', 'Depresión moderada', 'Normal']
)

# 5. Índice de masa corporal del bebé (adaptado)
df_selected['IMC_BEBE'] = df_selected['PESO (Gramos)'] / (df_selected['TALLA (CentImetros)']**2)

print("\nNuevas variables creadas:")
print("1. CATEGORIA_PESO: Categorización del peso al nacer")
print("2. CATEGORIA_GESTACION: Clasificación por tiempo de gestación")
print("3. RIESGO_APGAR1: Categorías de riesgo según APGAR a 1 minuto")
print("4. APGAR2_CAT: Categorización del APGAR a los 5 minutos (variable objetivo para clasificación)")
print("5. IMC_BEBE: Índice de masa corporal adaptado para el bebé")

# Mostrar distribución de las nuevas variables categóricas
for col in ['CATEGORIA_PESO', 'CATEGORIA_GESTACION', 'RIESGO_APGAR1', 'APGAR2_CAT']:
    print(f"\nDistribución de {col}:")
    print(df_selected[col].value_counts())

# ==============================================================================
# 3.7 ANÁLISIS DE CORRELACIONES PARA REDUNDANCIA
# ==============================================================================
print("\n" + "="*80)
print("3.7 ANÁLISIS DE CORRELACIONES PARA REDUNDANCIA")
print("="*80)

# Convertir variables categóricas a numéricas para análisis de correlación
df_corr = df_selected.copy()

# Usar one-hot encoding para variables categóricas
df_corr = pd.get_dummies(df_corr, drop_first=True)

# Calcular la matriz de correlación
corr_matrix = df_corr.corr()

# Visualizar matriz de correlación
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación')
plt.tight_layout()
plt.savefig('matriz_correlacion.png')
print("\nMatriz de correlación guardada como 'matriz_correlacion.png'")

# Identificar pares de variables con alta correlación (potencialmente redundantes)
correlation_threshold = 0.8

# Crear una matriz triangular superior para evitar duplicados
upper_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Encontrar pares con correlación alta
high_corr = [(corr_matrix.columns[i], corr_matrix.columns[j], upper_corr.iloc[i, j]) 
             for i in range(len(upper_corr.columns)) 
             for j in range(len(upper_corr.columns)) 
             if abs(upper_corr.iloc[i, j]) > correlation_threshold]

# Mostrar pares con alta correlación
print("\nPares de variables con alta correlación (potencialmente redundantes):")
if high_corr:
    for var1, var2, corr_val in high_corr:
        print(f"{var1} - {var2}: {corr_val:.4f}")
else:
    print("No se encontraron variables con correlación superior a 0.8")

# ==============================================================================
# 3.8 ANÁLISIS DE CORRELACIONES PARA IRRELEVANCIA (PREDICCIONES)
# ==============================================================================
print("\n" + "="*80)
print("3.8 ANÁLISIS DE CORRELACIONES PARA IRRELEVANCIA (PREDICCIONES)")
print("="*80)

# Analizar correlaciones con la variable objetivo (APGAR2)
target_correlations = corr_matrix['APGAR2'].sort_values(ascending=False)

print("\nCorrelaciones con APGAR2 (variable objetivo):")
print(target_correlations.head(10))
print("...")
print(target_correlations.tail(10))

# Visualizar las principales correlaciones con la variable objetivo
plt.figure(figsize=(14, 8))
target_correlations.drop('APGAR2').abs().sort_values(ascending=False).head(15).plot(kind='bar')
plt.title('Variables con mayor correlación absoluta con APGAR2')
plt.ylabel('Correlación absoluta')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('correlacion_con_objetivo.png')
print("\nGráfico de correlaciones con variable objetivo guardado como 'correlacion_con_objetivo.png'")

# Eliminar variables con baja correlación absoluta con el objetivo
low_correlation_threshold = 0.05
low_corr_vars = target_correlations.drop('APGAR2').index[
    abs(target_correlations.drop('APGAR2')) < low_correlation_threshold
]

print("\nVariables con baja correlación con APGAR2 (potencialmente irrelevantes):")
if len(low_corr_vars) > 0:
    for var in low_corr_vars:
        print(f"{var}: {target_correlations[var]:.4f}")
else:
    print("No se encontraron variables con correlación inferior a 0.05")

# ==============================================================================
# 3.9 REDUCCIÓN DE DIMENSIÓN (OPCIONAL EN PREDICCIONES)
# ==============================================================================
print("\n" + "="*80)
print("3.9 REDUCCIÓN DE DIMENSIÓN")
print("="*80)

try:
    # Seleccionar solo las variables numéricas para PCA
    # Excluir la variable objetivo y sus versiones categóricas
    cols_to_exclude = ['APGAR2']
    for col in df_corr.columns:
        if 'APGAR2_CAT' in col:
            cols_to_exclude.append(col)
            
    numeric_features = df_corr.select_dtypes(include=[np.number]).drop(columns=cols_to_exclude, errors='ignore')

    # Estandarizar las variables
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_features)

    # Aplicar PCA
    pca = PCA()
    pca_results = pca.fit_transform(scaled_features)

    # Visualizar la varianza explicada
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Explicada Acumulada')
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Varianza explicada')
    plt.title('Varianza Explicada por Componente')
    plt.legend()
    plt.grid(True)
    plt.savefig('pca_varianza_explicada.png')
    print("\nGráfico de varianza explicada por PCA guardado como 'pca_varianza_explicada.png'")

    # Número de componentes necesarios para explicar al menos el 80% de la varianza
    n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.8) + 1
    print(f"\nNúmero de componentes para explicar al menos el 80% de la varianza: {n_components}")
    
    # Visualización en 2D con PCA
    plt.figure(figsize=(12, 8))
    if 'APGAR2_CAT' in df_selected.columns:
        target = df_selected['APGAR2_CAT']
        target_num = pd.Categorical(target).codes
        
        # Graficar las dos primeras componentes
        scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=target_num, 
                            alpha=0.6, cmap='viridis', s=50)
        plt.title('PCA: Visualización de Datos en 2D por categoría de APGAR2')
        plt.colorbar(scatter, label='Categoría APGAR2')
    else:
        plt.scatter(pca_results[:, 0], pca_results[:, 1], alpha=0.6, s=50)
        plt.title('PCA: Visualización de Datos en 2D')
        
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid(True)
    plt.savefig('pca_visualizacion_2d.png')
    print("\nGráfico de visualización 2D con PCA guardado como 'pca_visualizacion_2d.png'")
except Exception as e:
    print(f"\nError en el análisis PCA: {e}")

# ==============================================================================
# 3.10 BALANCEO (CLASIFICACIÓN)
# ==============================================================================
print("\n" + "="*80)
print("3.10 BALANCEO (CLASIFICACIÓN)")
print("="*80)

# Verificar si las clases del APGAR2 están balanceadas
if 'APGAR2_CAT' in df_selected.columns:
    apgar_counts = df_selected['APGAR2_CAT'].value_counts()
    print("\nDistribución de clases de APGAR2_CAT:")
    print(apgar_counts)

    # Visualizar la distribución
    plt.figure(figsize=(10, 6))
    sns.barplot(x=apgar_counts.index, y=apgar_counts.values)
    plt.title('Distribución de Clases de APGAR2')
    plt.ylabel('Conteo')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.savefig('distribucion_clases_apgar2.png')
    print("\nGráfico de distribución de clases guardado como 'distribucion_clases_apgar2.png'")

    # Evaluar si hay desbalance
    min_class = apgar_counts.min()
    max_class = apgar_counts.max()
    ratio = min_class / max_class
    
    print(f"\nRatio clase minoritaria/mayoritaria: {ratio:.4f}")
    if ratio < 0.2:
        print("El dataset está muy desbalanceado. Se recomienda aplicar técnicas de balanceo.")
    elif ratio < 0.5:
        print("El dataset está moderadamente desbalanceado. Considerar técnicas de balanceo.")
    else:
        print("El dataset está relativamente balanceado. No es imprescindible aplicar balanceo.")
    
    print("\nNOTA: Para aplicar técnicas de balanceo (SMOTE, RandomUnderSampler, etc.) en la fase de modelado,")
    print("se recomienda instalar la biblioteca 'imbalanced-learn' con: pip install imbalanced-learn")
else:
    print("\nLa variable categorizada APGAR2_CAT no fue encontrada en el dataset.")

# ==============================================================================
# 3.11 TRANSFORMACIONES
# ==============================================================================
print("\n" + "="*80)
print("3.11 TRANSFORMACIONES")
print("="*80)

# Normalización de variables numéricas
# Seleccionar columnas numéricas para normalización
numeric_cols = df_selected.select_dtypes(include=[np.number]).columns

# Crear una copia del dataframe para la normalización
df_normalized = df_selected.copy()

# Aplicar StandardScaler (Z-score normalization)
scaler = StandardScaler()
df_normalized[numeric_cols] = scaler.fit_transform(df_selected[numeric_cols])

# Verificar el resultado de la normalización
print("\nEstadísticas descriptivas después de la normalización:")
normalized_stats = df_normalized[numeric_cols].describe().T[['mean', 'std', 'min', 'max']]
print(normalized_stats)

# ==============================================================================
# Guardar los datasets procesados
# ==============================================================================
print("\n" + "="*80)
print("GUARDAR DATASETS PROCESADOS")
print("="*80)

# Guardar el dataset procesado
df_selected.to_csv('datos_procesados.csv', index=False)
df_normalized.to_csv('datos_normalizados.csv', index=False)
print("\nDatasets procesados guardados en:")
print("- 'datos_procesados.csv': Dataset limpio sin normalización")
print("- 'datos_normalizados.csv': Dataset limpio con variables numéricas normalizadas")

# ==============================================================================
# Conclusiones
# ==============================================================================
print("\n" + "="*80)
print("CONCLUSIONES")
print("="*80)

print("""
En este script se ha realizado una limpieza completa de los datos para predecir
el APGAR score a los 5 minutos. Las transformaciones y limpieza realizadas incluyen:

1. Integración de datos y selección de variables relevantes
2. Descripción estadística completa
3. Tratamiento de valores atípicos
4. Manejo de valores nulos
5. Creación de nuevas variables predictivas
6. Análisis de correlaciones para identificar redundancia e irrelevancia
7. Reducción de dimensionalidad para visualización
8. Análisis de balance de clases
9. Normalización de variables

Los datos procesados están listos para ser utilizados en la fase de modelado,
ya sea para clasificación o regresión según el análisis realizado.
""")

print("="*80)
print("Limpieza de datos completada exitosamente!")
print("="*80) 