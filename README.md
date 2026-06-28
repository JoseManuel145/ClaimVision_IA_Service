# ClaimVision IA Service

API de inferencia de daños vehiculares utilizando un autoencoder (CNN) + K-Means.

## Stack

- **FastAPI** + **Uvicorn**
- **PostgreSQL** (Supabase) con **SQLAlchemy** asíncrono + **asyncpg**
- **PyTorch** (autoencoder) + **scikit-learn** (K-Means)
- Docker

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Estado del servicio |
| `GET` | `/api/v1/health` | Health check con estado de modelos |
| `POST` | `/api/v1/predict` | Predecir daño desde una imagen |
| `GET` | `/api/v1/history` | Historial paginado de inferencias |
| `POST` | `/api/v1/retrain` | Re-entrenar K-Means con nuevas imágenes |

Documentación interactiva en `/docs` (Swagger) o `/redoc`.

## Requisitos

- Python 3.11+
- PostgreSQL (o Supabase)
- Modelos en `models/`:
  - `encoder_best.pth` + `encoder_config.json`
  - `kmeans.pkl`
  - `cluster_mapping.json`

## Configuración

Copiar `.env.example` a `.env` y ajustar variables:

```env
DATABASE_URL=postgresql+asyncpg://postgres:password@host:5432/postgres
MODELS_DIR=models
LOG_LEVEL=INFO
ORIGINS=["*"]
```

### Base de datos

Ejecutar la migración en tu base de datos (Supabase SQL Editor o psql):

```bash
psql "$DATABASE_URL" -f migrations/001_create_inferences.sql
```

## Ejecutar local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Iniciar servidor
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker

```bash
docker compose up --build
```

Solo incluye el servicio `api` (la base de datos se conecta a Supabase externo).

## Estructura

```
app/
├── application/          # Casos de uso
├── domain/               # Modelos de dominio y puertos
├── infra/
│   ├── db/               # SQLAlchemy + repositorio
│   ├── ml/               # Encoder y clustering
│   └── mapping/          # Mapeo clúster → tipo de daño
└── presentation/         # Rutas, schemas y dependencias
migrations/               # Scripts SQL
models/                   # Modelos ML (no trackeados)
```
