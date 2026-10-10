# Clasificador de gatos y perros con EfficientNetB0

> **Estado:** la auditoría del dataset y el EDA están ejecutados; el entrenamiento completo sigue pendiente. Aún no hay métricas finales ni un modelo entrenado guardado.

Proyecto reproducible en español para auditar el conjunto de imágenes, explorar sus características y entrenar un clasificador binario con TensorFlow/Keras. El modelo usa **EfficientNetB0 sin pesos preentrenados**, con el backbone entrenable desde el inicio, entradas RGB de **200 × 200** y dos salidas softmax.

## Inicio rápido

1. Prepara el entorno según la sección [Entorno](#entorno).
2. Coloca los archivos `cat.<id>.jpg` y `dog.<id>.jpg` en `data/raw/dogs-vs-cats/train/`.
3. Desde la raíz, ejecuta `python src/app.py audit` y revisa el notebook `src/explore.ipynb`.
4. Para entrenar el modelo y obtener métricas finales, ejecuta `python src/app.py train --batch-size 16 --epochs 30 --seed 42`. En CPU puede tardar muchas horas.

## Datos y procedencia

Coloca las imágenes etiquetadas localmente en `data/raw/dogs-vs-cats/train/`, con nombres `cat.<id>.jpg` o `dog.<id>.jpg`. El directorio raw está excluido de Git porque contiene datos voluminosos y/o sujetos a licencias. Este repositorio no descarga datos automáticamente. La página de la competición original es [Kaggle Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats/data); la descarga puede requerir cuenta y aceptación de reglas.

La auditoría local detecta nombres desconocidos, archivos no decodificables, tamaños/modos, hashes SHA-256 y duplicados exactos. Genera `data/processed/splits.csv`, `audit.json` y `excluded_files.csv`. El manifiesto usa semilla 42, asignación aproximadamente estratificada 70/15/15, clases explícitas `cat=0` y `dog=1`, y agrupa duplicados SHA-256 para evitar que la misma imagen cruce particiones. Si cambias el dataset, vuelve a ejecutar la auditoría.

## Entorno

Se verificó con Python 3.13 y TensorFlow 2.20.0 en CPU. Se recomienda entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorFlow 2.20.0 está fijado para reproducibilidad y compatibilidad con el Python observado. Un backend GPU no es requisito, pero entrenar el modelo completo en CPU puede tomar bastante tiempo; no reduzcas la arquitectura ni reportes una corrida abreviada como resultado final.

## Flujo

Desde la raíz del repositorio:

```bash
# Auditar, deduplicar por grupo y crear las particiones reproducibles
python src/app.py audit

# Ejecutar el EDA reproducible, con salidas almacenadas en el notebook
# (abrir src/explore.ipynb en Jupyter/VS Code y ejecutar todas las celdas)

# Entrenar con validación para selección y test reservado para evaluación final
python src/app.py train --batch-size 16 --epochs 30 --seed 42

# Inferencia de una imagen
python src/app.py predict data/raw/dogs-vs-cats/train/cat.0.jpg
```

También existe la tarea de VS Code **Run dog-cat model training**. El entrenamiento puede tardar muchas horas en CPU; usa GPU compatible cuando esté disponible o lanza el comando en una sesión persistente y supervisada.

La arquitectura es `EfficientNetB0(include_top=False, weights=None, input_shape=(200, 200, 3))`, backbone entrenable, `GlobalAveragePooling2D`, `Dense(128, relu)` y `Dense(2, softmax)`. Usa etiquetas enteras y `sparse_categorical_crossentropy`. La augmentación aleatoria está integrada solo en la ruta de entrenamiento. EfficientNetB0 de Keras incorpora su propia capa de reescalado; el pipeline entrega píxeles RGB de 0–255 y no añade un segundo reescalado.

`ModelCheckpoint` y `EarlyStopping` seleccionan por `val_loss`. El conjunto de prueba no participa en ajuste ni selección y se consulta al final una sola vez. La evaluación completa queda guardada como métricas de test en `models/metadata.json`; solo deben compartirse resultados generados por una ejecución real.

## Artefactos

- `models/dogs_cats_efficientnetb0.keras`: modelo completo seleccionado.
- `models/metadata.json`: arquitectura, configuración, historial de validación y métricas finales de test.
- `models/history.json`: curvas registradas por época.
- `models/reload_verification.json`: comprobación de equivalencia de predicción al recargar.
- `data/processed/splits.csv`, `audit.json`, `excluded_files.csv`: manifiesto reproducible y evidencia de auditoría.

El modelo puede regenerarse con los datos, dependencias y semilla documentados; el artefacto `.keras` y los archivos de datos brutos se excluyen de Git.
