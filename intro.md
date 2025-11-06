# Procesamiento del Lenguaje Natural: Jarvis Calling Hiring Contest

Este proyecto aborda el problema de clasificación de texto propuesto en el **Jarvis Calling Hiring Contest**, mediante la implementación y comparación de diferentes enfoques de **Deep Learning y Machine Learning** aplicados al **Procesamiento del Lenguaje Natural (NLP)**.  

El objetivo principal es analizar el rendimiento de múltiples modelos representativos de distintas familias de arquitecturas, evaluando su capacidad para capturar patrones lingüísticos y semánticos en los datos.

---

## Metodología general

El desarrollo del proyecto se estructura en cuatro etapas principales:

### 1. Análisis Exploratorio de Datos (EDA)
Se realiza un estudio preliminar del conjunto de datos siguiendo las recomendaciones del artículo *Complete Guide to EDA on Text Data*.  
El análisis incluye:
- Distribución de clases y longitud de los textos.  
- Frecuencia de palabras y visualización de *word clouds* y *n-gramas*.  
- Exploración de patrones léxicos y semánticos relevantes.

### 2. Preprocesamiento del texto
Se implementan técnicas de limpieza y normalización, incluyendo:
- Eliminación de símbolos, stopwords y caracteres no alfabéticos.  
- Tokenización y secuenciación adaptada al modelo utilizado.  
- Uso de *embeddings* preentrenados en los casos correspondientes (*Word2Vec*, *FastText*).

### 3. Implementación de modelos
Se desarrollan y entrenan cinco modelos distintos para comparar su desempeño:

| # | Modelo | Descripción breve |
|---|--------|-------------------|
| 1 | **DistilBERT / RoBERTa-base (fine-tuned)** | Ajuste fino de un *Transformer* preentrenado. |
| 2 | **Word2Vec + BiLSTM** | Modelo secuencial con *embeddings* y memoria a largo plazo. |
| 3 | **CNN-1D para texto** | Extracción de características locales mediante convoluciones. |
| 4 | **TF-IDF + XGBoost (GPU)** | Enfoque clásico de representación y clasificación. |
| 5 | **FastText** | Entrenamiento local con *embeddings* subpalabra optimizados. |

Cada modelo se entrena y evalúa bajo las mismas condiciones experimentales, con conjuntos de entrenamiento, validación y prueba definidos previamente.  
*(No se aplicó búsqueda sistemática de hiperparámetros por GridSearch debido a limitaciones de tiempo y recursos.)*

### 4. Evaluación y comparación
El desempeño de cada modelo se evalúa mediante las siguientes métricas:

- **Exactitud (Accuracy)**  
- **Precisión (Precision)**  
- **Exhaustividad (Recall)**  
- **F1-Score (Macro)**  
- **Matriz de confusión**  
- **ROC-AUC (One-vs-Rest, si aplica)**  

Adicionalmente, se registran tiempos de entrenamiento y uso de GPU en los modelos basados en *Transformers* para evaluar su costo computacional.

---

## Estructura del libro

El contenido del proyecto está organizado en las siguientes secciones:

1. **EDA** — Análisis exploratorio del conjunto de datos.  
2. **DistilBERT** — Fine-tuning de modelo *Transformer*.  
3. **TF Models** — Implementaciones de BiLSTM, CNN-1D y FastText.  
4. **Resultados** — Evaluación comparativa y análisis crítico de desempeño.

---

```{tableofcontents}
