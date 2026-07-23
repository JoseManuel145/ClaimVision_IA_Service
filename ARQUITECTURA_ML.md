# ClaimVision IA Service — Arquitectura del Backend de Machine Learning

**Version:** 3.5.0  
**Stack:** Python 3.11 · FastAPI · PyTorch · scikit-learn · Tesseract · Groq Cloud  
**Base de datos:** PostgreSQL (async via SQLAlchemy + asyncpg)

---

## 1. Arquitectura General

### 1.1 Diagrama del Sistema

```
                        ClaimVision_IAService
                        =====================
                                |
    ┌───────────────────────────┼───────────────────────────┐
    │                           │                           │
    │   FastAPI (21 endpoints)  │                           │
    │                           │                           │
    └───┬───────────┬───────────┼───────────┬───────────────┘
        │           │           │           │
        ▼           ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │   OCR   │ │Supervi- │ │No Super-│ │   NLP   │
   │         │ │  sado   │ │ visado  │ │         │
   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
        │           │           │           │
        ▼           ▼           ▼           ▼
   Tesseract   ResNet18    Encoder CNN   Groq Cloud
   PyMuPDF     (PyTorch)   + K-Means    ├─ LLM
   Regex       pHash       (sklearn)     └─ Whisper
```

### 1.2 Clean Architecture (Hexagonal)

Cada módulo sigue la misma estructura de capas:

```
modulo/
├── domain/          ← Modelos de negocio + interfaces (Protocol)
│   ├── models.py        Dataclasses puras, sin dependencias
│   └── ports.py         Protocolos (interfaces) que define el dominio
├── application/     ← Casos de uso (orquestan lógica)
│   └── *_use_case.py    Cada caso de uso = una operación del negocio
├── infra/           ← Implementaciones concretas (adapters)
│   ├── ml/              Modelos de ML, preprocesamiento
│   ├── db/              Repositorios PostgreSQL
│   └── ...              Servicios externos (Groq, Tesseract, etc.)
└── presentation/    ← Capa HTTP (FastAPI)
    ├── routes.py        Endpoints
    ├── schemas.py       Request/Response schemas (Pydantic)
    └── dependencies.py  Inyección de dependencias
```

**Principio clave:** El dominio (domain/) no depende de ninguna框架 externa. Las interfaces (Protocol) definen qué debe hacer cada servicio, y la infraestructura las implementa. Esto permite cambiar implementaciones (ej: Ollama → Groq) sin modificar casos de uso ni rutas.

### 1.3 Stack Tecnológico

| Capa | Tecnología | Propósito |
|------|-----------|-----------|
| Web Framework | FastAPI 0.138 | API REST async |
| ORM | SQLAlchemy 2.0 (async) | Mapeo objeto-relacional |
| Database Driver | asyncpg | Conexión async a PostgreSQL |
| Deep Learning | PyTorch 2.13 | ResNet18, Encoder CNN |
| Clustering | scikit-learn 1.9 | K-Means |
| OCR | Tesseract (pytesseract) | Reconocimiento óptico de caracteres |
| PDF | PyMuPDF 1.28 | Extracción de texto/imágenes de PDF |
| STT | Groq Cloud (Whisper V3 Turbo) | Transcripción de audio |
| LLM | Groq Cloud (Llama 3.1 8B) | Extracción de entidades de daño |
| Hashing | imagehash (pHash) | Deduplicación de imágenes |
| Server | Uvicorn | ASGI server |

---

## 2. Módulo OCR — Extracción y Parsing de Documentos

**Prefix:** `/api/v1/ocr`  
**Objetivo:** Extraer texto de PDFs (pólizas) e imágenes/PDFs (credencial INE), parsear campos estructurados con regex, y validar cruzadamente póliza contra INE.

### 2.1 Pipeline de Extracción PDF (3 pasos)

```
PDF de entrada
    │
    ▼
┌─────────────────────────┐
│ Paso 1: Extracción      │  PyMuPDF extrae texto embebido
│ directa de texto        │  ¿Tiene ≥20 caracteres? → Devolver
└────────────┬────────────┘
             │ No
             ▼
┌─────────────────────────┐
│ Paso 2: OCR de imágenes │  Extrae imágenes incrustadas del PDF
│ incrustadas             │  OCR cada imagen con Tesseract
└────────────┬────────────┘
             │ No hay suficiente texto
             ▼
┌─────────────────────────┐
│ Paso 3: Render + OCR    │  Renderiza página completa a 400 DPI
│ completo                │  OCR con 2 métodos: simple vs mejorado
│                         │  (2x upscale + sharpen + contraste + B&W)
│                         │  Selecciona el mejor por detección MRZ
└─────────────────────────┘
```

### 2.2 Extracción Regex — INE (Credencial Electoral)

El archivo `regex_document_extractor.py` (1019 líneas) implementa extracción pura con regex, sin dependencia de LLM. Pipeline:

```
OCR Text → Normalizar → Segmentar → Clasificar → Extraer
```

**Métodos principales:**

| Campo | Estrategia | Regex/Validación |
|-------|-----------|-----------------|
| **CURP** | 3 estrategias: label, loose pattern, collapsed-whitespace | `[A-Z]{4}\d{6}[HM][A-Z]{2}[A-Z]{3}[A-Z0-9]{2}` (18 chars) |
| **Clave Elector** | 4 estrategias + corrección posicional OCR | 18 caracteres alfanuméricos |
| **Nombre** | 3 niveles: MRZ → label "NOMBRE" → heurística (2-5 palabras) | Sin dígitos, sin labels conocidos |
| **Domicilio** | 3 niveles: label DOMICILIO → extracción por CP → scoring semántico | |
| **RFC** | Pattern + validación | `[A-Z]{3,4}\d{6}[A-Z0-9]{2,3}` |
| **Fecha Nacimiento** | Label date + fallback a CURP | Derivada de posición 6-11 de CURP |
| ** Sexo ** | Label + CURP posición 10 + MRZ | H/M/F |
| **MRZ TD1** | Parser de líneas MRZ del reverso INE | IDMEX, YYMMDD, H/M |

**Corrección de errores OCR:** Caracteres confundidos: `O→0, I→1, S→5, Z→2, G→6, B→8, L→1` en posiciones numéricas.

### 2.3 Extracción Regex — Póliza de Seguro

| Campo | Estrategia |
|-------|-----------|
| Número de póliza | Label "NUMERO DE POLIZA" o patrón cercano a "POLIZA" |
| Aseguradora | Línea después de "Apoderado" |
| Nombre asegurado | Label "Nombre/Name" + fallback |
| Marca/Modelo/Placas | Patrones bilingües (MARCA/MAKE, MODELO/MODEL, etc.) |
| VIN | Label "VIN" o "SERIE" |
| Vigencia | "DESDE"/"HASTA" con abreviaturas de meses |
| Año | Label "ANO"/"YEAR" |

### 2.4 Validación Cruzada (Póliza vs INE)

```
ExtractPoliza + ExtractIne
         │
         ▼
┌─────────────────────────┐
│ 1. Nombre match         │  Normalizar acentos, eliminar stop-words
│                         │  ¿≥2 tokens en común? → Match
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 2. CURP/RFC consistente │  Primeros 10 chars de CURP vs RFC
│                         │  Deben coincidir (4 letras + 6 dígitos)
└─────────────────────────┘
```

### 2.5 Validación de Imagen

Validación específica por tipo de documento:

| Parámetro | INE | Póliza |
|-----------|-----|--------|
| Resolución mínima | 800×500 px | 1000×700 px |
| Aspect ratio | ~0.631 / 1.585 | ~0.773 / 1.294 |
| Nitidez (Laplacian) | > 50 | > 30 |
| Brillo | 50-220 | 60-220 |

---

## 3. Módulo Supervisado — Clasificación con ResNet18

**Prefix:** `/api/v2`  
**Objetivo:** Clasificar imágenes de daño vehicular en 6 categorías usando transfer learning con ResNet18.

### 3.1 Arquitectura del Modelo

```
Imagen de entrada (RGB)
    │
    ▼
┌─────────────────────────┐
│ Preprocesamiento        │  Resize(256) → CenterCrop(224)
│                         │  ToTensor → Normalize(ImageNet)
│                         │  Output: [1, 3, 224, 224]
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ ResNet18 (pretrained)   │  torch.hub.load('pytorch/vision', 'resnet18')
│ ImageNet weights        │  Features: Conv → BN → ReLU × 4 bloques
│                         │  AvgPool → Flatten → 512-dim
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ FC Layer (custom)       │  Linear(512, 6)
│                         │  6 clases de daño
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Softmax + Threshold     │  confidence < 0.35 → "Sin Dano"
│                         │  confidence ≥ 0.35 → Clase predicha
└─────────────────────────┘
```

### 3.2 Clases de Daño

| ID | Clase | Descripción |
|----|-------|-------------|
| 0 | `Rotura_Cristal` | Vidrio roto |
| 1 | `Rayadura` | Rayón superficial |
| 2 | `Abolladura` | Abolladura en chasis |
| 3 | `Grietas` | Grietas en superficie |
| 4 | `Neumatico_pinchado` | Neumático pinchado |
| 5 | `Faro_roto` | Faro dañado |

### 3.3 Mapeo de Severidad

La severidad se calcula **inversamente** a la confianza del modelo:

```
confianza ≥ 0.8  →  Severidad "Bajo"   (el modelo está muy seguro = daño menor/clasificable)
confianza ≥ 0.5  →  Severidad "Medio"
confianza < 0.5  →  Severidad "Alto"   (el modelo no está seguro = daño severo/ambiguo)
confianza < 0.35 →  "Sin Dano"         (por debajo del umbral mínimo)
```

**Lógica:** Si el modelo no puede clasificar bien el daño (baja confianza), es porque el daño es más severo o complejo.

### 3.4 Pipeline de Reentrenamiento

```
Dataset (imágenes + labels.json)
    │
    ▼
┌─────────────────────────┐
│ Split 80/20             │  StratifiedShuffleSplit
│ (estratificado)         │  Mantiene proporción de clases
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Data Augmentation       │  Resize(256), RandomFlip(H/V),
│ (solo train)            │  Rotation(15°), Affine, ColorJitter,
│                         │  CenterCrop(224)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Entrenamiento           │  Loss: CrossEntropyLoss (pesos por frecuencia)
│ (background thread)     │  Optim: Adam(lr)
│                         │  Scheduler: ReduceLROnPlateau(patience=3)
│                         │  Early Stopping: patience=5
│                         │  Hasta 100 épocas
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Guardado atómico        │  os.replace() → classifier_best.pth
│                         │  Actualiza class_mapping.json
│                         │  Actualiza DB job con progreso
└─────────────────────────┘
```

**Características del reentrenamiento:**
- Ejecuta en `loop.run_in_executor(None, _train)` (thread pool, no bloquea el event loop)
- Pesos de clase inversamente proporcionales a la frecuencia (maneja desbalanceo)
- Progreso actualizado en DB vía `asyncio.run_coroutine_threadsafe`
- Guardado atómico con `os.replace()` (nunca deja archivo corrupto)

### 3.5 Deduplicación por pHash

El endpoint `/predict-all` acepta hasta 50 imágenes y deduplica automáticamente:

```
Imágenes de entrada
    │
    ▼
┌─────────────────────────┐
│ Calcular pHash          │  imagehash.phash() → hash de 64 bits
│ cada imagen             │  (perceptual hash, tolerante a
│                         │  rotación/escala/compresión)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Comparar distancias     │  hamming distance ≤ 5 → Duplicado
│ Hamming                 │  hamming distance > 5 → Única
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Clasificar solo únicas  │  Solo las imágenes únicas pasan
│                         │  por ResNet18 (ahorra inferencia)
└─────────────────────────┘
```

### 3.6 Matriz de Costos (MXN)

| Tipo de Daño | Bajo | Medio | Alto |
|--------------|------|-------|------|
| Rotura Cristal | $3,000 | $8,000 | $15,000 |
| Rayadura | $500 | $2,000 | $5,000 |
| Abolladura | $1,000 | $5,000 | $10,000 |
| Grietas | $2,000 | $6,000 | $12,000 |
| Neumático Pinchado | $1,500 | $3,000 | $5,000 |
| Faro Roto | $2,000 | $5,000 | $10,000 |

---

## 4. Módulo No Supervisado — Encoder + K-Means

**Prefix:** `/api/v1`  
**Objetivo:** Clasificar imágenes de daño sin etiquetas usando un encoder convolucional + clustering K-Means.

### 4.1 Arquitectura del Encoder

```
Imagen de entrada (RGB, 128×128)
    │
    ▼
┌──────────────────────────────────────────────┐
│ Bloque Conv 1: Conv2d(3→32) + BN + ReLU     │  128×128 → 64×64
│ Bloque Conv 2: Conv2d(32→64) + BN + ReLU    │  64×64 → 32×32
│ Bloque Conv 3: Conv2d(64→128) + BN + ReLU   │  32×32 → 16×16
│ Bloque Conv 4: Conv2d(128→256) + BN + ReLU  │  16×16 → 8×8
└────────────┬─────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────┐
│ Flatten: 256 × 8 × 8 = 16,384              │
│ FC1: 16,384 → 512 (ReLU)                    │
│ FC2: 512 → 128 (latent dimension)           │
└────────────┬─────────────────────────────────┘
             │
             ▼
    Vector latente 128-dim
    [0.23, -0.45, 0.89, ..., 0.12]
```

**Configuración del encoder** (`encoder_config.json`):
- `latent_dim`: 128
- `img_size`: 128
- `channels`: [32, 64, 128, 256]
- `kernel_size`: 3, `stride`: 2, `padding`: 1

### 4.2 Pipeline de Predicción

```
Imagen → Preprocesar → Encoder → Vector 128-dim → K-Means → Cluster ID
                                                                      │
                                                              ┌───────┘
                                                              ▼
                                                      Cluster Mapping
                                                              │
                                                              ▼
                                                    tipo_dano + severidad
                                                    + confianza
```

### 4.3 K-Means Clustering

| Parámetro | Valor |
|-----------|-------|
| K (número de clusters) | 6 |
| n_init | 10 |
| random_state | 42 |
| Modelo guardado | `models/kmeans.pkl` (pickle) |

**Métricas de clustering:**
- Silhouette Score: 0.076 (bajo, esperado para imágenes de daño vehicular con alta variabilidad)
- Davies-Bouldin: 3.33
- Inertia: guardada en `clustering_metrics.json`

### 4.4 Mapeo de Clusters a Daños

| Cluster | tipo_dano | Label dominante | % dominante | Imágenes |
|---------|-----------|----------------|-------------|----------|
| 0 | Rotura_Cristal | Glass shatter | 30.4% | 1,083 |
| 1 | Rayadura | Scratch | 35.9% | 2,049 |
| 2 | Abolladura | Dent | 38.8% | 707 |
| 3 | Rayadura | Scratch | 40.7% | 1,151 |
| 4 | Rayadura | Scratch | 37.1% | 1,267 |
| 5 | Abolladura | Dent | 34.5% | 943 |

### 4.5 Severidad por Distancia al Centroid

```
ratio = distancia_al_centroid / distancia_maxima
ratio < 0.33  →  "Bajo"    (cerca del centroid = daño típico)
ratio < 0.66  →  "Medio"
ratio ≥ 0.66  →  "Alto"    (lejos del centroid = daño atípico)

confianza = 1.0 - ratio
```

---

## 5. Módulo NLP — Transcripción + Extracción de Entidades

**Prefix:** `/api/v1/nlp`  
**Objetivo:** Transcribir audio del conductor (español) a texto, luego extraer entidades de daño estructuradas usando un LLM en la nube.

### 5.1 Pipeline Completo

```
Audio del conductor (m4a/wav/mp3/etc.)
    │
    ▼
┌─────────────────────────┐
│ Groq Whisper V3 Turbo   │  Transcripción speech-to-text
│ (whisper-large-v3)      │  228x tiempo real
│                         │  $0.04/hora transcrita
└────────────┬────────────┘
             │
             ▼
    Texto en español:
    "El carro tiene pérdida de potencia
     en el motor, hace ruidos extraños..."
             │
             ▼
┌─────────────────────────┐
│ Groq Llama 3.1 8B       │  Extracción de entidades de daño
│ (llama-3.1-8b-instant)  │  840 tokens/segundo
│                         │  $0.05/M input, $0.08/M output
└────────────┬────────────┘
             │
             ▼
    JSON estructurado:
    {"d": [
      {"t": "perdida_potencia", "s": "Alto",
       "p": "motor", "x": "...", "c": 0.9},
      {"t": "ruidos_anormales", "s": "Medio",
       "p": "motor", "x": "...", "c": 0.8}
    ]}
             │
             ▼
┌─────────────────────────┐
│ Mapeo a DamageEntity    │  tipo_dano, severidad,
│                         │  parte_afectada, sintoma,
│                         │  confianza
└─────────────────────────┘
```

### 5.2 Optimización de Tokens

El endpoint `/analizar` optimiza el consumo de tokens Groq:

| Componente | Estrategia | Tokens |
|-----------|-----------|--------|
| System prompt | Instrucciones concisas, sin redundancia | ~60 |
| User prompt | Solo el texto del conductor (sin instrucciones) | ~20-50 |
| Response format | `json_object` mode (sin schema overhead) | 0 |
| Campos abreviados | `t,s,p,x,c` en vez de nombres largos | -50% output |
| **Total** | | **~200 input, ~180 output** |

**Costo por request:** ~$0.025 USD (vs ~$0.30 con el approach anterior de 1,386 tokens)

### 5.3 Jobs Asíncronos

La transcripción de audio soporta modo asíncrono (patrón job):

```
POST /nlp/transcribir (audio)
    │
    ├─→ 202 Accepted
    │   {job_id: "abc-123", status: "pending", progress: 0}
    │
    └─→ Background Task:
        10%  → status: "processing"
        60%  → STT completado
        90%  → LLM completado
        100% → status: "completed", result_id: "xyz-789"

GET /nlp/transcribir/status/{job_id}
    │
    └─→ {status, progress, result: {...}}
```

### 5.4 Comparativa Groq vs Local

| Aspecto | Ollama (antes) | Groq Cloud (ahora) |
|---------|---------------|-------------------|
| Modelo LLM | qwen2.5:3b (3B params) | llama-3.1-8b (8B params) |
| Velocidad | ~10-30 tokens/seg | 840 tokens/seg |
| STT modelo | faster-whisper tiny (39M) | whisper-large-v3-turbo |
| STT velocidad | ~10x real-time | 228x real-time |
| Latencia total | 5-15 segundos | 1-3 segundos |
| Infraestructura | Requiere servidor Ollama | Solo API key |
| Costo/mes (~100 req/día) | Gratis (CPU) | ~$2 USD |

---

## 6. Base de Datos

### 6.1 Diagrama de Tablas

```
┌──────────────────┐     ┌──────────────────────┐
│ ocr_documents    │     │ v2_predictions       │
│──────────────────│     │──────────────────────│
│ id (PK)          │     │ id (PK)              │
│ filename         │     │ filename             │
│ text             │     │ class_id             │
│ page_count       │     │ tipo_dano            │
│ created_at       │     │ severidad            │
└──────────────────┘     │ confianza            │
                         │ prob_dist (JSON)     │
┌──────────────────┐     │ created_at           │
│ v2_retrain_jobs  │     └──────────────────────┘
│──────────────────│
│ id (PK)          │     ┌──────────────────────┐
│ status           │     │ inferences           │
│ total_epochs     │     │──────────────────────│
│ current_epoch    │     │ id (PK)              │
│ best_accuracy    │     │ filename             │
│ loss_history     │     │ cluster_id           │
│ error            │     │ tipo_dano            │
│ created_at       │     │ severidad            │
│ completed_at     │     │ confianza            │
└──────────────────┘     │ distancia_centroide  │
                         │ created_at           │
┌──────────────────────┐ └──────────────────────┘
│ nlp_transcripciones │
│──────────────────────│ ┌──────────────────────┐
│ id (PK)              │ │ nlp_damage_entities  │
│ filename             │ │──────────────────────│
│ texto                │ │ id (PK)              │
│ duracion_seg         │ │ transcripcion_id (FK)│
│ created_at           │ │ tipo_dano            │
└──────┬───────────────┘ │ severidad            │
       │                 │ parte_afectada       │
       └────────────────→│ sintoma              │
                         │ confianza            │
┌──────────────────────┐ │ created_at           │
│ nlp_jobs             │ └──────────────────────┘
│──────────────────────│
│ id (PK)              │
│ filename             │
│ status               │
│ progress             │
│ result_id (FK → nlp_transcripciones)
│ error                │
│ created_at           │
│ updated_at           │
└──────────────────────┘
```

### 6.2 Resumen de Tablas

| Tabla | Módulo | Registros | Relaciones |
|-------|--------|-----------|------------|
| `ocr_documents` | OCR | 1 por request OCR | Independiente |
| `v2_predictions` | Supervisado | 1 por predicción | Independiente |
| `v2_retrain_jobs` | Supervisado | 1 por job de reentreno | Independiente |
| `inferences` | No Supervisado | 1 por predicción | Independiente |
| `nlp_transcripciones` | NLP | 1 por transcripción | Padre de damage_entities |
| `nlp_damage_entities` | NLP | N por transcripción | FK → nlp_transcripciones.id |
| `nlp_jobs` | NLP | 1 por job | FK → nlp_transcripciones.id |

**Total: 7 tablas**, todas con UUID string como PK, timestamps TIMESTAMPTZ con UTC.

---

## 7. Decisiones Arquitectónicas

### 7.1 ¿Por qué Clean Architecture?

- **Intercambiabilidad:** Cambiar Ollama por Groq requirió modificar solo `groq_extractor.py` + `dependencies.py`. Los use cases y routes no se tocaron.
- **Testeabilidad:** Cada protocolo puede tener mocks para tests.
- **Separación de responsabilidades:** El dominio no sabe de FastAPI, PyTorch, ni Groq.

### 7.2 ¿Por qué Async en todo?

- FastAPI es nativamente async. Las operaciones de DB (asyncpg) y las llamadas a Groq son I/O-bound.
- El reentrenamiento (CPU/GPU-bound) se ejecuta en `ThreadPoolExecutor` para no bloquear el event loop.
- Los jobs de transcrición usan `asyncio.create_task` para procesamiento en background.

### 7.3 ¿Por qué Groq Cloud en vez de Ollama?

| Factor | Ollama | Groq Cloud |
|--------|--------|-----------|
| Latencia | 5-15s | 1-3s |
| Velocidad | ~30 TPS | ~840 TPS |
| Infraestructura | Servidor dedicado (GPU) | Solo API key |
| Mantenimiento | Actualizar modelos, gestionar RAM | Nada |
| Costo mensual | Electricidad + hardware | ~$2 USD |
| Calidad | 3B params | 8B params |

### 7.4 ¿Por qué no LLM en OCR?

El parsing de documentos (INE, póliza) usa regex puro porque:
1. **Velocidad:** Regex es instantáneo, un LLM tomaría 2-5 segundos por documento.
2. **Costo:** Cada documento costaría tokens Groq. Con regex es gratis.
3. **Precisión:** Los campos de documentos oficiales tienen formatos fijos. Regex es más preciso que LLM para datos estructurados.
4. **Dependencia:** Sin conexión a internet = sin LLM. Regex funciona offline.

### 7.5 ¿Por qué dos enfoques de clasificación?

- **Supervisado (v2):** Cuando hay datos etiquetados. Más preciso, 6 clases definidas, confidence thresholding.
- **No supervisado (v1):** Cuando no hay etiquetas. Descubre patrones automáticamente. Útil para datos nuevos o exploración inicial.

### 7.6 Threshold de Confianza (0.35)

Si el modelo supervisado tiene confianza < 35%, retorna "Sin Dano" en vez de una clasificación errónea. Esto evita falsos positivos en producción.

### 7.7 Guardado Atómico de Modelos

```python
# Nunca sobreescribir un archivo directamente
tmp_path = model_path + ".tmp"
torch.save(state_dict, tmp_path)
os.replace(tmp_path, model_path)  # Atómico en POSIX
```

Si el proceso muere durante el guardado, el archivo original se mantiene intacto.

---

## 8. Endpoints de la API (21 total)

### OCR (5 endpoints)
| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/api/v1/ocr` | Extraer texto de PDF |
| GET | `/api/v1/ocr/history` | Historial de documentos OCR |
| POST | `/api/v1/ocr/extract-poliza` | Extraer datos de póliza |
| POST | `/api/v1/ocr/extract-ine` | Extraer datos de INE |
| POST | `/api/v1/ocr/extract-and-validate` | Extraer + validar cruzadamente |

### Supervisado (6 endpoints)
| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/api/v2/predict` | Clasificar imagen individual |
| POST | `/api/v2/predict-all` | Clasificar múltiples (con dedup) |
| POST | `/api/v2/obtener-resumen` | Estimación de costos |
| POST | `/api/v2/retrain` | Iniciar reentreno async |
| GET | `/api/v2/retrain/{job_id}` | Estado del reentreno |
| GET | `/api/v2/history` | Historial de predicciones |

### No Supervisado (4 endpoints)
| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/api/v1/predict` | Clasificar imagen (encoder+K-Means) |
| GET | `/api/v1/history` | Historial de inferencias |
| POST | `/api/v1/retrain` | Re-entrenar K-Means |
| GET | `/api/v1/health` | Estado del servicio |

### NLP (5 endpoints)
| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/api/v1/nlp/transcribir` | Transcribir audio (async) |
| GET | `/api/v1/nlp/transcribir/status/{job_id}` | Estado de transcripción |
| POST | `/api/v1/nlp/analizar` | Analizar texto directo |
| GET | `/api/v1/nlp/history` | Historial de transcripciones |
| GET | `/api/v1/nlp/{id}` | Detalle de transcripción |

---

## 9. Artefactos de Modelo

| Archivo | Formato | Tamaño aprox. | Propósito |
|---------|---------|---------------|-----------|
| `classifier_best.pth` | PyTorch state dict | ~44 MB | Pesos ResNet18 |
| `class_mapping.json` | JSON | <1 KB | Mapeo ID → nombre de clase |
| `encoder_best.pth` | PyTorch state dict | ~2 MB | Pesos encoder CNN |
| `encoder_config.json` | JSON | <1 KB | Config arquitectura encoder |
| `kmeans.pkl` | Pickle (sklearn) | ~50 KB | Modelo K-Means |
| `cluster_mapping.json` | JSON | ~2 KB | Mapeo cluster → daño |
| `damage_matrix.json` | JSON | <1 KB | Matriz de costos MXN |
| `clustering_metrics.json` | JSON | <1 KB | Métricas de clustering |

---

## 10. Resumen para Defensa

### Puntos Clave del Proyecto

1. **Arquitectura limpia:** Hexagonal con 4 módulos independientes, fácil de mantener y escalar.

2. **Enfoque dual de clasificación:** Supervisado (ResNet18) para datos etiquetados + No supervisado (Encoder+K-Means) para datos sin etiquetas.

3. **Procesamiento de documentos sin LLM:** Regex inteligente con corrección de errores OCR, parsing de MRZ, y validación cruzada póliza-INE.

4. **NLP en la nube:** Groq Cloud ofrece inferencia 40x más rápida que alternativas locales, a costo mínimo (~$2/mes).

5. **Reentrenamiento en producción:** Los modelos se actualizan con nuevos datos sin downtime, con early stopping y guardado atómico.

6. **Deduplicación inteligente:** pHash evita procesar imágenes duplicadas, ahorrando tiempo de inferencia.

7. **Jobs asíncronos:** Operaciones pesadas (transcripción, reentreno) no bloquean la API.

8. **Producción real:** Docker Compose con gateway (nginx), backend, IA service, y PostgreSQL. Health checks, logs rotativos, y restart automático.
