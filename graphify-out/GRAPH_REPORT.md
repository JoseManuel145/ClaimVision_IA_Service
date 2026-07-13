# Graph Report - .  (2026-07-13)

## Corpus Check
- Corpus is ~19,827 words - fits in a single context window. You may not need a graph.

## Summary
- 606 nodes · 1351 edges · 68 communities (63 shown, 5 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 101 edges (avg confidence: 0.64)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- OCR Extraction Pipeline
- Docs & API Schemas
- Unsupervised ML Engine
- NLP Domain & Tests
- Unsupervised Use Cases
- NLP Transcription Jobs
- Unsupervised Routes
- NLP Routes
- Supervised History & Models
- NLP LLM & Whisper Infra
- Unsupervised DB & Inference
- Supervised Predict Pipeline
- NLP History Use Case
- NLP Transcription Use Case
- Supervised DB & Retrain
- Core Config & Database
- Supervised Routes
- NLP DB Repository
- Unsupervised Tests
- Supervised Classifier Ports
- Unsupervised Predict Flow
- Supervised Tests
- Unsupervised History Flow
- Unsupervised Retrain Flow
- Unsupervised Encoder Ports
- Supervised Route Tests
- Supervised Retrain Use Case
- Unsupervised Predict Tests
- OCR Route Tests
- Shell Scripts
- Model Update Script

## God Nodes (most connected - your core abstractions)
1. `VozTranscripcion` - 34 edges
2. `TranscripcionJob` - 32 edges
3. `DamageEntity` - 25 edges
4. `TranscripcionJobUseCase` - 23 edges
5. `PostgresTranscripcionJobRepository` - 22 edges
6. `PostgresVozRepository` - 21 edges
7. `ClaimVision IA Service` - 20 edges
8. `OCRDocument` - 17 edges
9. `RetrainJob` - 17 edges
10. `TranscribirUseCase` - 15 edges

## Surprising Connections (you probably didn't know these)
- `pytesseract` --semantically_similar_to--> `Autoencoder CNN`  [INFERRED] [semantically similar]
  requirements.txt → README.md
- `umap-learn` --semantically_similar_to--> `K-Means Clustering`  [INFERRED] [semantically similar]
  requirements.txt → README.md
- `faster-whisper` --semantically_similar_to--> `Autoencoder CNN`  [INFERRED] [semantically similar]
  requirements.txt → README.md
- `TestAnalizarRoute` --uses--> `VozTranscripcion`  [INFERRED]
  tests/nlp/test_routes.py → app/modules/nlp/domain/models.py
- `TestDetailRoute` --uses--> `VozTranscripcion`  [INFERRED]
  tests/nlp/test_routes.py → app/modules/nlp/domain/models.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **ML Damage Prediction Pipeline** — readme_autoencoder_cnn, readme_kmeans, readme_cluster_mapping_json, readme_encoder_best_pth, readme_kmeans_pkl, readme_predict_endpoint [EXTRACTED 1.00]
- **REST API Surface** — readme_predict_endpoint, readme_history_endpoint, readme_retrain_endpoint, readme_health_endpoint, docs_api_predict_response, docs_api_history_response, docs_api_retrain_response, docs_api_health_response [EXTRACTED 1.00]
- **Clean Architecture Layers** — readme_clean_architecture, readme_fastapi, readme_postgresql_supabase, readme_sqlalchemy_async, readme_autoencoder_cnn, readme_kmeans [INFERRED 0.85]

## Communities (68 total, 5 thin omitted)

### Community 0 - "OCR Extraction Pipeline"
Cohesion: 0.05
Nodes (50): ExtractAndValidateUseCase, ExtractIneUseCase, ExtractPolizaUseCase, OcrUseCase, DocumentExtraction, IneData, OCRDocument, PolizaData (+42 more)

### Community 1 - "Docs & API Schemas"
Cohesion: 0.09
Nodes (41): Docker Compose Configuration, API Service (Container), ollama_data Volume, Ollama Service (Container), HealthResponse Schema, HistoryItem Schema, HistoryResponse Schema, HTTPValidationError Schema (+33 more)

### Community 2 - "Unsupervised ML Engine"
Cohesion: 0.07
Nodes (18): Encoder, Tensor, _build_model(), ImageFolderDataset, _load_class_names(), callable, Dataset, Module (+10 more)

### Community 3 - "NLP Domain & Tests"
Cohesion: 0.08
Nodes (11): DamageEntity, mock_llm_service(), mock_voz_repository(), mock_job_use_case(), mock_llm(), now(), TestAnalizarRoute, TestDetailRoute (+3 more)

### Community 4 - "Unsupervised Use Cases"
Cohesion: 0.12
Nodes (8): ClusteringService, ClusterMapper, ImagePreprocessor, InferenceRepository, Protocol, Tensor, TorchImagePreprocessor, get_preprocessor()

### Community 5 - "NLP Transcription Jobs"
Cohesion: 0.25
Nodes (9): TranscripcionJobUseCase, TranscripcionJob, TranscripcionJobRepository, PostgresTranscripcionJobRepository, test_get_job_status_returns_job(), test_process_completes_successfully(), test_process_marks_failed_on_llm_error(), test_process_marks_failed_on_stt_error() (+1 more)

### Community 6 - "Unsupervised Routes"
Cohesion: 0.19
Nodes (16): SklearnClusteringService, get_clustering(), health(), history(), predict(), HistoryUseCase, UploadFile, retrain() (+8 more)

### Community 7 - "NLP Routes"
Cohesion: 0.27
Nodes (17): analizar(), _build_transcripcion_response(), nlp_detail(), nlp_history(), HistoryUseCase, UploadFile, transcribir(), transcribir_status() (+9 more)

### Community 8 - "Supervised History & Models"
Cohesion: 0.18
Nodes (8): V2HistoryUseCase, V2Prediction, V2PredictionRepository, get_v2_history_use_case(), mock_retrain_job_repo(), mock_v2_prediction_repo(), test_v2_history_empty(), test_v2_history_execute()

### Community 9 - "NLP LLM & Whisper Infra"
Cohesion: 0.17
Nodes (11): OllamaExtractor, WhisperSTTService, get_history_use_case(), get_llm_service(), get_nlp_repository(), get_stt_service(), get_transcribir_use_case(), get_transcripcion_job_use_case() (+3 more)

### Community 10 - "Unsupervised DB & Inference"
Cohesion: 0.15
Nodes (9): Inference, PostgresInferenceRepository, AsyncSession, InferenceTable, Base, get_repository(), AsyncSession, mock_inference_repo() (+1 more)

### Community 11 - "Supervised Predict Pipeline"
Cohesion: 0.18
Nodes (11): V2PredictUseCase, Tensor, SupervisedPreprocessor, get_v2_predict_use_case(), get_v2_prediction_repository(), get_v2_preprocessor(), get_v2_retrain_job_repository(), AsyncSession (+3 more)

### Community 12 - "NLP History Use Case"
Cohesion: 0.24
Nodes (7): HistoryUseCase, VozTranscripcion, VozRepository, test_history_get_by_id(), test_history_get_by_id_not_found(), test_history_list_paginated(), test_history_list_paginated_empty()

### Community 13 - "NLP Transcription Use Case"
Cohesion: 0.23
Nodes (10): TranscribirUseCase, AsyncSession, NlpAnalysisService, Protocol, SpeechToTextService, async_sessionmaker, test_transcribir_execute_success(), test_transcribir_pipeline_order() (+2 more)

### Community 14 - "Supervised DB & Retrain"
Cohesion: 0.23
Nodes (7): RetrainJob, PostgresRetrainJobRepository, PostgresV2PredictionRepository, AsyncSession, Base, V2PredictionTable, V2RetrainJobTable

### Community 15 - "Core Config & Database"
Cohesion: 0.17
Nodes (11): Config, Settings, get_session(), init_db(), AsyncSession, lifespan(), BaseSettings, FastAPI (+3 more)

### Community 16 - "Supervised Routes"
Cohesion: 0.29
Nodes (13): health_v2(), history_v2(), predict_v2(), UploadFile, retrain_status(), retrain_v2(), BaseModel, V2HealthResponse (+5 more)

### Community 17 - "NLP DB Repository"
Cohesion: 0.31
Nodes (6): PostgresVozRepository, AsyncSession, NlpDamageEntityTable, NlpJobTable, NlpTranscripcionTable, Base

### Community 18 - "Unsupervised Tests"
Cohesion: 0.15
Nodes (5): get_predict_use_case(), mock_services(), TestHealthRoute, TestHistoryRoute, TestRetrainRoute

### Community 19 - "Supervised Classifier Ports"
Cohesion: 0.17
Nodes (4): ClassifierService, callable, Protocol, RetrainJobRepository

### Community 20 - "Unsupervised Predict Flow"
Cohesion: 0.32
Nodes (7): PredictUseCase, PredictionResult, test_predict_calls_all_steps_in_order(), test_predict_propagates_preprocessor_failure(), test_predict_success(), test_predict_empty_image(), test_predict_success()

### Community 21 - "Supervised Tests"
Cohesion: 0.17
Nodes (4): get_v2_retrain_use_case(), mock_retrain_use_case(), TestHealthRoute, TestHistoryRoute

### Community 22 - "Unsupervised History Flow"
Cohesion: 0.31
Nodes (7): HistoryUseCase, PaginatedResult, get_history_use_case(), HistoryUseCase, test_history_use_case_empty(), test_history_use_case_execute(), test_history_use_case_pagination()

### Community 23 - "Unsupervised Retrain Flow"
Cohesion: 0.40
Nodes (5): RetrainUseCase, TrainingMetrics, get_retrain_use_case(), test_retrain_less_images_than_k(), test_retrain_success()

### Community 24 - "Unsupervised Encoder Ports"
Cohesion: 0.31
Nodes (5): EncoderService, JsonClusterMapper, TorchEncoderService, get_encoder(), get_mapper()

### Community 26 - "Supervised Retrain Use Case"
Cohesion: 0.48
Nodes (4): V2RetrainUseCase, test_get_job_status_not_found(), test_get_job_status_returns_job(), test_start_retrain_creates_job_and_returns()

## Knowledge Gaps
- **4 isolated node(s):** `Config`, `run.sh script`, `update_models.sh script`, `scikit-learn`
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TranscripcionJobUseCase` connect `NLP Transcription Jobs` to `NLP Routes`, `NLP LLM & Whisper Infra`, `NLP History Use Case`, `NLP Transcription Use Case`, `NLP DB Repository`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Why does `get_session()` connect `Core Config & Database` to `Unsupervised Encoder Ports`, `NLP LLM & Whisper Infra`, `Supervised Predict Pipeline`, `OCR Extraction Pipeline`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Why does `TranscripcionJob` connect `NLP Transcription Jobs` to `NLP DB Repository`, `NLP Domain & Tests`, `NLP History Use Case`, `NLP Transcription Use Case`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `VozTranscripcion` (e.g. with `HistoryUseCase` and `TranscribirUseCase`) actually correct?**
  _`VozTranscripcion` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `TranscripcionJob` (e.g. with `TranscripcionJobUseCase` and `NlpAnalysisService`) actually correct?**
  _`TranscripcionJob` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `DamageEntity` (e.g. with `NlpAnalysisService` and `SpeechToTextService`) actually correct?**
  _`DamageEntity` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `TranscripcionJobUseCase` (e.g. with `TranscripcionJob` and `VozTranscripcion`) actually correct?**
  _`TranscripcionJobUseCase` has 6 INFERRED edges - model-reasoned connections that need verification._