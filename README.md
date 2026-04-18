# Manhwa & Manga AI Assistant (RAG + IA)

## Descripción del Proyecto

Este proyecto consiste en el desarrollo de un asistente inteligente basado en técnicas de **LLM (Modelos de Lenguaje)** y **RAG (Retrieval-Augmented Generation)**, orientado a la consulta de información sobre **mangas y manhwas**.

El sistema permite a los usuarios realizar preguntas en lenguaje natural y obtener respuestas contextualizadas, incluyendo:

* Descripción de obras
* Traducción automática al español
* Identificación del tipo (manga/manhwa)
* Valoración (score)
* Visualización de imagen

---

## Objetivo

Mejorar la experiencia de usuario en plataformas de lectura digital mediante un sistema capaz de interpretar consultas y entregar información relevante de forma automática.

---

## Tecnologías Utilizadas

* **Frontend:** HTML + JavaScript
* **Backend:** Python (Flask)
* **IA (Local):** MarianMT (transformers)
* **API externa:** Jikan API (MyAnimeList)
* **Arquitectura:** RAG (fuente interna + externa)

---

## Arquitectura del Sistema

```
Usuario
   ↓
Frontend (Interfaz Chat)
   ↓
Backend (Flask API)
   ↓
Motor RAG
   ↓
├── Base de Datos (JSON - manhwas)
├── API Jikan (mangas)
   ↓
Modelo de Traducción (IA)
   ↓
Generación de Respuesta
   ↓
Usuario
```

---

## Instalación y Ejecución

### Clonar repositorio

```bash
git clone https://github.com/cristopherRamirezU/MANHWA-RAG-AI
cd manhwa-rag-ai/backend
```

---

###  Crear entorno virtual (opcional pero recomendado)

```bash
python -m venv venv
venv\Scripts\activate
```

---

### Instalar dependencias

```bash
pip install flask flask-cors requests transformers torch
```

---

### Ejecutar backend

```bash
python app.py
```

Deberías ver:

```
Running on http://127.0.0.1:5000
```

---

### Ejecutar frontend

Abre el archivo:

```
frontend/index.html
```

Puedes usar:

* Live Server (VSCode)
* o abrir directamente en navegador

---

## Cómo usar la aplicación

### Ejemplos de preguntas

#### Información general

* `Naruto`
* `Solo Leveling`
* `de que se trata berserk`

---

####  Valoración

* `valoracion Naruto`
* `que nota tiene solo leveling`
* `puntuacion attack on titan`

---

#### Consulta libre

* `explicame naruto`
* `informacion de solo leveling`
* `de que trata one piece`

---

## Funcionalidades del Sistema

✔ Búsqueda en base de datos interna (JSON)
✔ Consulta a API externa (Jikan)
✔ Limpieza de preguntas (extracción de nombre)
✔ Traducción automática al español (IA local)
✔ Detección de intención (valoración vs descripción)
✔ Respuesta estructurada
✔ Visualización de imagen

---

## Ejemplo de Respuesta

```
Según nuestra información, el manga "Naruto" tiene una valoración de 8.9.

Descripción:
[Naruto traducido al español]

Imagen:
[URL de imagen]
```

---

## Consideraciones

* La API Jikan puede tener límites de uso
* La primera ejecución descarga el modelo de traducción (puede tardar)
* El sistema funciona sin necesidad de API Key

---

## Posibles Mejoras

* Recomendaciones automáticas por género
* Historial de usuario
* Implementación de base de datos vectorial (FAISS)
* Integración con más APIs

---

## Autor

**Cristopher Alexander Ramirez Ubilla**
Ingeniería de Soluciones con IA

---

## Notas

Este proyecto fue desarrollado con fines académicos, aplicando conceptos de:

* LLM
* RAG
* Arquitectura de sistemas IA

---

