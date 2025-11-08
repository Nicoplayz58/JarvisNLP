# 🧠 Procesamiento del Lenguaje Natural — Jarvis Calling Hiring Contest

Este proyecto aborda un desafío de **clasificación de texto** planteado en el *Jarvis Calling Hiring Contest*, mediante la implementación y comparación de diferentes enfoques de **Machine Learning** y **Deep Learning** aplicados al **Procesamiento del Lenguaje Natural (NLP)**.

El objetivo principal es evaluar la capacidad de distintos modelos para capturar patrones lingüísticos y semánticos en los datos, analizando su rendimiento en términos de precisión, generalización y eficiencia computacional.

---

## 🧩 Metodología general

El desarrollo del proyecto se estructura en **cuatro etapas principales**:

### 1️⃣ Análisis Exploratorio de Datos (EDA)
Basado en las recomendaciones del artículo *Complete Guide to EDA on Text Data*, se realiza un estudio preliminar del conjunto de datos que incluye:

- Distribución de clases y longitud de textos.  
- Frecuencia de palabras, *word clouds* y análisis de *n-gramas*.  
- Exploración de patrones léxicos y semánticos relevantes.

### 2️⃣ Preprocesamiento del texto
Se implementan técnicas de limpieza y normalización del corpus:

- Eliminación de símbolos, *stopwords* y caracteres no alfabéticos.  
- Tokenización y secuenciación adaptada a cada arquitectura.  
- Uso de *embeddings* preentrenados (Word2Vec, FastText).

### 3️⃣ Implementación de modelos
Se entrenan y comparan **cinco modelos representativos** de distintas familias de arquitecturas:

| Nº | Modelo | Descripción breve |
|----|---------|-------------------|
| 1 | **DistilBERT / RoBERTa-base (fine-tuned)** | Ajuste fino de un Transformer preentrenado. |
| 2 | **Word2Vec + BiLSTM** | Modelo secuencial con memoria a largo plazo. |
| 3 | **CNN-1D para texto** | Extracción de características locales mediante convoluciones. |
| 4 | **TF-IDF + XGBoost (GPU)** | Representación clásica combinada con aprendizaje de gradiente. |
| 5 | **FastText** | Entrenamiento local con embeddings de subpalabras. |

> ⚙️ Todos los modelos fueron entrenados con conjuntos de entrenamiento, validación y prueba predefinidos.  
> No se aplicó *GridSearch* sistemático por limitaciones de tiempo y recursos.

### 4️⃣ Evaluación y comparación
Se evalúan los modelos con métricas estándar de clasificación:

- **Accuracy**, **Precision**, **Recall**, **F1-Score (Macro)**  
- **Matriz de confusión**  
- **ROC-AUC (One-vs-Rest)**  
- **Tiempos de entrenamiento y uso de GPU**

---

## 📚 Estructura del Jupyter Book

El proyecto está documentado y publicado como un **Jupyter Book** con las siguientes secciones:

- `EDA/` — Análisis exploratorio del conjunto de datos  
- `DistilBERT/` — Ajuste fino del modelo Transformer  
- `TF Models/` — Implementaciones de BiLSTM, CNN-1D y FastText  
- `Resultados/` — Evaluación comparativa y análisis crítico  

👉 **Versión en línea:**  
🔗 [https://nicoplayz58.github.io/JarvisNLP/](https://nicoplayz58.github.io/JarvisNLP/)

---

## 🛠️ Tecnologías utilizadas

- **Python 3.12**  
- **Jupyter Book**  
- **scikit-learn**, **TensorFlow**, **PyTorch**  
- **Transformers (Hugging Face)**  
- **XGBoost (GPU)**  
- **FastText**, **Gensim**, **NLTK**, **spaCy**

---

## 📦 Estructura del repositorio

