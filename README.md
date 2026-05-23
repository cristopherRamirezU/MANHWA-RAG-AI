# ManhwaBot — Agente RAG con Memoria y Planificación

> Asistente conversacional inteligente para consultas sobre manhwa, manga y manhua.  
> Arquitectura de agente **ReAct** (Reasoning + Acting) con memoria de corto y largo plazo.

**Curso:** ISY0101 — Ingeniería de Soluciones con IA  
**Autores:** Cristopher Alexander Ramírez Ubilla  
**Versión:** 2.0 — Agente Funcional

---

## Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Componentes Implementados](#componentes-implementados)
4. [Decisiones de Diseño](#decisiones-de-diseño)
5. [Instalación paso a paso](#instalación-paso-a-paso)
6. [Cómo usar la aplicación](#cómo-usar-la-aplicación)
7. [Límites del sistema](#límites-del-sistema)
8. [API Endpoints](#api-endpoints)
9. [Flujos de Trabajo](#flujos-de-trabajo)
10. [Estructura del Repositorio](#estructura-del-repositorio)
11. [Referencias Bibliográficas](#referencias-bibliográficas)

---

## Descripción General

ManhwaBot es un **agente funcional** que integra herramientas de consulta, escritura y razonamiento para responder preguntas sobre contenido de lectura digital. El sistema utiliza el patrón **ReAct** (Yao et al., 2022) para combinar razonamiento en lenguaje natural con la invocación autónoma de herramientas, sin depender de frameworks externos de agentes.

### Capacidades del agente

| Capacidad | Descripción |
|---|---|
| Consulta semántica | Recupera información con embeddings multilingües |
| Búsqueda externa | Consulta la API Jikan (MyAnimeList) |
| Recomendaciones | Filtra títulos por género o tema |
| Memoria de sesión | Mantiene contexto de las últimas 5 interacciones |
| Memoria persistente | Guarda historial y preferencias del usuario en disco |
| Planificación | Decide autónomamente qué herramienta usar en cada paso |
| Control de dominio | Rechaza preguntas fuera del ámbito manhwa/manga/manhua |

---

## Arquitectura del Sistema

### Diagrama general de orquestación

```mermaid
graph TD
    U([Usuario]) -->|Pregunta en lenguaje natural| FE[Frontend HTML/JS]
    FE -->|POST /agent| API[Flask API]

    API --> AGT[Agente ReAct\nGroq SDK]

    AGT -->|Planifica y decide| LOOP{Ciclo ReAct}

    LOOP -->|Thought| LLM[LLM Groq\nLlama-3.3-70b]
    LLM -->|Action| T1[buscar_manhwa]
    LLM -->|Action| T2[recomendar_por_genero]
    LLM -->|Action| T3[ver_historial]
    LLM -->|Action| T4[guardar_preferencia]

    T1 -->|1. Semántica| SS[Motor Semántico\nsentence-transformers]
    T1 -->|2. Fallback| JK[API Jikan\nMyAnimeList]
    T2 --> DB[(BD Local\nmanhwas.json)]
    SS --> DB

    T3 --> MM[MemoryManager]
    T4 --> MM

    MM --> SHT[Memoria Corto Plazo\ndeque — ventana k=5]
    MM --> LHT[(Memoria Largo Plazo\nJSON por usuario)]

    LOOP -->|Observation| LLM
    LLM -->|Final Answer| API
    API --> FE
    FE --> U

    style AGT fill:#4a90d9,color:#fff
    style MM fill:#e8a838,color:#fff
    style SS fill:#50b86c,color:#fff
    style LLM fill:#9b59b6,color:#fff
```

### Diagrama de capas

```
┌─────────────────────────────────────────────────────────────┐
│                     CAPA DE PRESENTACIÓN                     │
│              frontend/index.html  (HTML + JS)                │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP POST /agent
┌─────────────────────────▼───────────────────────────────────┐
│                      CAPA DE API                             │
│                  backend/app.py  (Flask)                     │
│        Endpoints:  /chat (legacy)  /agent  /health          │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    CAPA DE AGENTE                            │
│               backend/agent.py  (Groq SDK — ReAct)          │
│   LLM: Groq Llama-3.3-70b  │  Prompt ReAct en español      │
│   Ciclo: Thought → Action → Observation → Final Answer      │
└──────────┬──────────────────────────────────┬───────────────┘
           │                                  │
┌──────────▼──────────┐             ┌─────────▼──────────────┐
│    CAPA DE TOOLS    │             │    CAPA DE MEMORIA      │
│   backend/tools.py  │             │ backend/memory_manager  │
│                     │             │                         │
│ • buscar_manhwa     │             │ Corto plazo:            │
│ • recomendar_por_   │             │  deque(maxlen=5)        │
│   genero            │             │  en RAM por sesión      │
│ • ver_historial     │             │                         │
│ • guardar_prefe-    │             │ Largo plazo:            │
│   rencia            │             │  JSON por usuario       │
└──────────┬──────────┘             │  data/sessions/         │
           │                        └─────────────────────────┘
┌──────────▼──────────┐
│ CAPA DE RECUPERACIÓN│
│ semantic_search.py  │
│ rag_pipeline.py     │
│ API Jikan           │
│ BD Local JSON       │
└─────────────────────┘
```

---

## Componentes Implementados

### 1. Agente ReAct — `backend/agent.py`

Implementa el patrón **ReAct** (Reasoning + Acting) mediante parsing de texto. En cada turno el LLM genera un bloque estructurado que el sistema parsea para ejecutar herramientas:

```
Thought: El usuario pregunta por un manhwa de terror. Uso recomendar_por_genero.
Action: recomendar_por_genero
Action Input: Terror
                        ← sistema inyecta la observación aquí
Observation: • Sweet Home (Score: 8.4) — Hyun Cha queda atrapado...
Thought: Tengo suficiente información.
Final Answer: Te recomiendo Sweet Home...
```

### 2. Herramientas del Agente — `backend/tools.py`

| Herramienta | Función | Fuente de datos |
|---|---|---|
| `buscar_manhwa` | Información detallada de un título | BD local + Jikan API |
| `recomendar_por_genero` | Lista de títulos por género | BD local |
| `ver_historial` | Historial de lectura del usuario | Memoria LP (JSON) |
| `guardar_preferencia` | Registra una calificación | Memoria LP (JSON) |

### 3. Memoria — `backend/memory_manager.py`

**Corto plazo** (`deque` con `maxlen=5`):
- Ventana deslizante de las últimas 5 interacciones almacenada en RAM.
- Se incluye automáticamente en cada invocación del agente como historial.
- Al superar 5 turnos descarta el más antiguo automáticamente.

**Largo plazo** (JSON persistente en `backend/data/sessions/{user_id}.json`):
- Guarda preferencias con calificación y fecha.
- Guarda historial de títulos consultados.
- Persiste entre sesiones; no se pierde al reiniciar el servidor.

### 4. Búsqueda Semántica — `backend/semantic_search.py`

Recuperación de contexto semántico (IE4) usando embeddings:

- **Modelo**: `paraphrase-multilingual-MiniLM-L12-v2` (soporta español).
- **Proceso**: genera embeddings del corpus al inicio; calcula similitud coseno contra la query.
- **Threshold**: descarta resultados con similitud < 0.25 para evitar falsos positivos.
- **Ventaja**: encuentra títulos aunque la pregunta no use las palabras exactas del título.

### 5. Pipeline RAG Legado — `backend/rag_pipeline.py`

Mantiene el endpoint `/chat` original. Usa MarianMT para traducción EN→ES con carga diferida (no bloquea el inicio del servidor).

---

## Decisiones de Diseño

### ¿Por qué Groq SDK directo en vez de LangChain Agents?

Durante el desarrollo se probaron `create_react_agent` y `create_tool_calling_agent` de LangChain, pero presentaron incompatibilidades con Python 3.14 y con `llama-3.3-70b-versatile` (el modelo generaba function calls en formato propietario `<function=...>` en vez del estándar JSON). La solución fue implementar el loop ReAct directamente sobre el SDK de Groq, ganando control total del ciclo y eliminando dependencias frágiles.

### ¿Por qué Groq + Llama-3.3-70b?

Groq ofrece una API gratuita con latencia extremadamente baja gracias a su hardware LPU. Llama-3.3-70b tiene capacidades de razonamiento en español comparables a GPT-4o sin costo de API (Meta AI, 2024; GroqCloud, 2024).

### ¿Por qué el patrón ReAct por texto?

ReAct (Yao et al., 2022) permite que el LLM **planifique explícitamente** cada paso. Al usar parsing de texto en vez de function calling, el sistema es compatible con cualquier modelo y la cadena Thought→Action→Observation es completamente auditable en los logs del servidor.

### ¿Por qué sentence-transformers para búsqueda semántica?

`paraphrase-multilingual-MiniLM-L12-v2` es un modelo ligero (~120 MB) con soporte nativo para español. Permite recuperar contexto relevante sin coincidencia exacta de palabras. FAISS fue descartado por añadir complejidad innecesaria dado el tamaño de la BD (Reimers & Gurevych, 2019).

### ¿Por qué dos tipos de memoria?

Sigue la distinción cognitiva entre memoria de trabajo y memoria episódica a largo plazo (Tulving, 1972). El `deque` de corto plazo evita que el contexto crezca y agote el context window del LLM, mientras que el JSON persistente permite recuperar preferencias entre sesiones sin requerir una base de datos relacional.

---

## Instalación paso a paso

### Requisitos previos

- **Python 3.10 o superior** — descargar en [python.org/downloads](https://www.python.org/downloads/)
  > Al instalar marcar obligatoriamente **"Add Python to PATH"**
- **API Key de Groq gratuita** — registrarse en [console.groq.com](https://console.groq.com)

### 1. Clonar el repositorio

```bash
git clone https://github.com/cristopherRamirezU/MANHWA-RAG-AI
cd MANHWA-RAG-AI
```

### 2. Crear entorno virtual (recomendado)

```bash
# Crear
python -m venv venv

# Activar en Windows
venv\Scripts\activate

# Activar en macOS / Linux
source venv/bin/activate
```

Cuando el entorno está activo verás `(venv)` al inicio del prompt.

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> La primera instalación puede tardar varios minutos por el tamaño de `torch` y `sentence-transformers`.

### 4. Configurar la API Key

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Abrir el archivo `.env` y reemplazar el valor:

```
GROQ_API_KEY=gsk_tu_clave_real_aqui
```

### 5. Iniciar el servidor

```bash
cd backend
python app.py
```

Deberías ver:

```
* Running on http://127.0.0.1:5000
* Debugger is active!
```

### 6. Abrir el frontend

Abrir el archivo `frontend/index.html` en el navegador.  
En VS Code: clic derecho → **Open with Live Server**.

---

## Cómo usar la aplicación

### Preguntas sobre manhwa (flujo normal)

**Ejemplo 1 — Consulta de información:**
```
Tú:     ¿De qué trata Solo Leveling?
Bot:    Solo Leveling sigue la historia de Sung Jin-Woo, considerado
        el cazador más débil de la humanidad, que obtiene un sistema
        único que le permite subir de nivel sin límite...
```

**Ejemplo 2 — Recomendación por género:**
```
Tú:     Recomiéndame manhwas de terror
Bot:    Te recomiendo Sweet Home (Score: 8.4): Hyun Cha queda atrapado
        en su apartamento mientras el mundo colapsa y los humanos
        se convierten en monstruos que reflejan sus deseos más oscuros.
```

**Ejemplo 3 — Búsqueda sin título exacto (semántica):**
```
Tú:     Busco algo de un chico débil que se vuelve muy poderoso
Bot:    Basándome en tu descripción, te recomiendo Solo Leveling...
```

**Ejemplo 4 — Guardar favorito:**
```
Tú:     Me encantó Tower of God, le doy un 9
Bot:    Preferencia guardada: 'Tower of God' con calificación 9/10.
```

**Ejemplo 5 — Ver historial:**
```
Tú:     ¿Qué tengo en mi historial?
Bot:    Historial de lectura:
          - Tower of God (2026-05-23)
        Calificaciones guardadas:
          - Tower of God: 9/10
```

---

### Pregunta fuera del dominio (límite del agente)

El agente está restringido a manhwa, manga y manhua. Si se pregunta sobre otro tema, el sistema lo detecta y responde con un mensaje de redirección **sin consultar ninguna herramienta**:

```
Tú:     ¿Cuáles son los mejores animes shonen?
Bot:    Solo puedo ayudarte con manhwa, manga y manhua.
        ¿Te interesa que busque algo relacionado?
```

> Este comportamiento demuestra la toma de decisiones adaptativa del agente (IE6):  
> el LLM evalúa la pregunta, detecta que está fuera de dominio y responde directamente  
> con `Final Answer` sin invocar ninguna herramienta.

---

## Límites del sistema

| Tipo | Límite | Detalle |
|---|---|---|
| **Groq API — requests/min** | 30 por minuto | Se reinicia automáticamente cada 60 s |
| **Groq API — tokens/min** | 6,000 | Incluye pregunta + respuesta |
| **Groq API — requests/día** | 14,400 | Plan gratuito |
| **Agente — iteraciones** | 6 por pregunta | Si no llega a `Final Answer` en 6 pasos devuelve mensaje de error |
| **Agente — tokens/respuesta** | 1,024 | Configurable en `agent.py` → `max_tokens` |
| **Memoria corto plazo** | 5 interacciones | Ventana deslizante; el turno más antiguo se descarta |
| **BD local** | 10 títulos | Si no encuentra localmente consulta Jikan (miles de títulos) |

---

## API Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del servidor |
| `POST` | `/chat` | Pipeline RAG legado (sin agente) |
| `POST` | `/agent` | Agente ReAct con memoria |

### Ejemplo de petición

```bash
curl -X POST http://127.0.0.1:5000/agent \
  -H "Content-Type: application/json" \
  -d "{\"pregunta\": \"Recomiendame manhwas de terror\", \"user_id\": \"usuario1\"}"
```

### Respuesta

```json
{
  "respuesta": "Te recomiendo Sweet Home (Score: 8.4)...",
  "user_id": "usuario1"
}
```

---

## Flujos de Trabajo

### Flujo ReAct completo (IE5)

```
Pregunta del usuario
        │
        ▼
LLM genera ──► Thought: razona qué herramienta necesita
        │
        ▼
        ├── Action + Action Input ──► Sistema ejecuta herramienta
        │                                      │
        │         Observation ◄────────────────┘
        │
        ▼
LLM evalúa resultado
        │
        ├── necesita más info ──► nuevo ciclo (máx. 6 iteraciones)
        │
        └── suficiente info  ──► Final Answer → usuario
```

### Flujo de memoria (IE3)

```
Cada interacción
        │
        ├─► Corto Plazo (RAM):
        │     deque.append({human, ai})
        │     Si len > 5 → descarta el más antiguo automáticamente
        │
        └─► Largo Plazo (disco):
              Solo cuando se usa guardar_preferencia
              Escribe en data/sessions/{user_id}.json
              { "preferencias": [...], "historial": [...] }
```

### Flujo de búsqueda semántica (IE4)

```
Query del usuario
        │
        ▼
Embedding con MiniLM (paraphrase-multilingual)
        │
        ▼
Similitud coseno contra todos los embeddings del corpus
        │
        ├── similitud ≥ 0.25 → devuelve top-3 locales
        │
        └── sin resultados  → fallback a API Jikan (MyAnimeList)
```

---

## Estructura del Repositorio

```
MANHWA-RAG-AI/
├── backend/
│   ├── app.py                  API Flask (/chat, /agent, /health)
│   ├── agent.py                Agente ReAct con Groq SDK
│   ├── tools.py                Herramientas del agente (IE1)
│   ├── memory_manager.py       Memoria corto y largo plazo (IE3)
│   ├── semantic_search.py      Búsqueda semántica con embeddings (IE4)
│   ├── rag_pipeline.py         Pipeline RAG legado
│   └── data/
│       ├── manhwas.json        Base de datos local (10 títulos)
│       └── sessions/           Memorias LP por usuario — en .gitignore
├── frontend/
│   └── index.html              Interfaz web de chat
├── tests/
│   └── test_agent.py           Demostración de toma de decisiones (IE6)
├── .env.example                Plantilla de API Key — NO incluir .env real
├── .gitignore                  Excluye .env, venv, __pycache__, .claude/, etc.
├── requirements.txt            Dependencias del proyecto
└── README.md                   Este archivo
```

---

## Referencias Bibliográficas

Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., & Wang, H. (2023). *Retrieval-augmented generation for large language models: A survey*. arXiv preprint arXiv:2312.10997. https://arxiv.org/abs/2312.10997

GroqCloud. (2024). *Groq API documentation*. Groq Inc. https://console.groq.com/docs

Meta AI. (2024). *Llama 3: The most capable openly available LLM to date*. Meta Platforms Inc. https://llama.meta.com/llama3/

Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence embeddings using Siamese BERT-networks*. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. https://arxiv.org/abs/1908.10084

Tulving, E. (1972). Episodic and semantic memory. En E. Tulving & W. Donaldson (Eds.), *Organization of memory* (pp. 381–403). Academic Press.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). *ReAct: Synergizing reasoning and acting in language models*. arXiv preprint arXiv:2210.03629. https://arxiv.org/abs/2210.03629
