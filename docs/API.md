---
title: ClaimVision IA Service v1.0.0
language_tabs:
  - python: Python
language_clients:
  - python: ""
toc_footers: []
includes: []
search: true
highlight_theme: darkula
headingLevel: 2

---

<!-- Generator: Widdershins v4.0.1 -->

<h1 id="claimvision-ia-service">ClaimVision IA Service v1.0.0</h1>

> Scroll down for code samples, example requests and responses. Select a language for code samples from the tabs above or the mobile navigation menu.

<h1 id="claimvision-ia-service-default">Default</h1>

## Predecir tipo de daño vehicular

<a id="opIdpredict_api_v1_predict_post"></a>

> Code samples

```python
import requests
headers = {
  'Content-Type': 'multipart/form-data',
  'Accept': 'application/json'
}

r = requests.post('/api/v1/predict', headers = headers)

print(r.json())

```

`POST /api/v1/predict`

Recibe una imagen de daño vehicular, la procesa a través del encoder y K-Means, y devuelve el tipo de daño, severidad y confianza.

> Body parameter

```yaml
file: string

```

<h3 id="predecir-tipo-de-daño-vehicular-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|body|body|[Body_predict_api_v1_predict_post](#schemabody_predict_api_v1_predict_post)|true|none|

> Example responses

> 200 Response

```json
{
  "id": "string",
  "filename": "string",
  "tipo_dano": "string",
  "severidad": "string",
  "confianza": 0,
  "distancia_centroide": 0,
  "created_at": "string"
}
```

<h3 id="predecir-tipo-de-daño-vehicular-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|[PredictResponse](#schemapredictresponse)|
|422|[Unprocessable Entity](https://tools.ietf.org/html/rfc2518#section-10.3)|Validation Error|[HTTPValidationError](#schemahttpvalidationerror)|

<aside class="success">
This operation does not require authentication
</aside>

## Consultar historial de inferencias

<a id="opIdhistory_api_v1_history_get"></a>

> Code samples

```python
import requests
headers = {
  'Accept': 'application/json'
}

r = requests.get('/api/v1/history', headers = headers)

print(r.json())

```

`GET /api/v1/history`

Devuelve una lista paginada de todas las inferencias realizadas.

<h3 id="consultar-historial-de-inferencias-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|page|query|integer|false|Número de página|
|limit|query|integer|false|Elementos por página|

> Example responses

> 200 Response

```json
{
  "data": [
    {
      "id": "string",
      "filename": "string",
      "cluster_id": 0,
      "tipo_dano": "string",
      "severidad": "string",
      "confianza": 0,
      "distancia_centroide": 0,
      "created_at": "string"
    }
  ],
  "total": 0,
  "page": 0,
  "limit": 0
}
```

<h3 id="consultar-historial-de-inferencias-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|[HistoryResponse](#schemahistoryresponse)|
|422|[Unprocessable Entity](https://tools.ietf.org/html/rfc2518#section-10.3)|Validation Error|[HTTPValidationError](#schemahttpvalidationerror)|

<aside class="success">
This operation does not require authentication
</aside>

## Re-entrenar K-Means

<a id="opIdretrain_api_v1_retrain_post"></a>

> Code samples

```python
import requests
headers = {
  'Content-Type': 'multipart/form-data',
  'Accept': 'application/json'
}

r = requests.post('/api/v1/retrain', headers = headers)

print(r.json())

```

`POST /api/v1/retrain`

[Admin] Recibe un conjunto de imágenes y re-entrena el modelo K-Means. Requiere al menos K imágenes.

> Body parameter

```yaml
k: 2
files:
  - string

```

<h3 id="re-entrenar-k-means-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|body|body|[Body_retrain_api_v1_retrain_post](#schemabody_retrain_api_v1_retrain_post)|true|none|

> Example responses

> 200 Response

```json
{
  "k": 0,
  "silhouette": 0,
  "davies_bouldin": 0,
  "inertia": 0,
  "mapping": [
    {}
  ],
  "trained_at": "string"
}
```

<h3 id="re-entrenar-k-means-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|[RetrainResponse](#schemaretrainresponse)|
|422|[Unprocessable Entity](https://tools.ietf.org/html/rfc2518#section-10.3)|Validation Error|[HTTPValidationError](#schemahttpvalidationerror)|

<aside class="success">
This operation does not require authentication
</aside>

## Verificar estado del servicio

<a id="opIdhealth_api_v1_health_get"></a>

> Code samples

```python
import requests
headers = {
  'Accept': 'application/json'
}

r = requests.get('/api/v1/health', headers = headers)

print(r.json())

```

`GET /api/v1/health`

Indica si los modelos están cargados correctamente.

> Example responses

> 200 Response

```json
{
  "status": "string",
  "model_loaded": true,
  "k_value": 0
}
```

<h3 id="verificar-estado-del-servicio-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|[HealthResponse](#schemahealthresponse)|

<aside class="success">
This operation does not require authentication
</aside>

<h1 id="claimvision-ia-service-root">Root</h1>

## Root

<a id="opIdroot__get"></a>

> Code samples

```python
import requests
headers = {
  'Accept': 'application/json'
}

r = requests.get('/', headers = headers)

print(r.json())

```

`GET /`

> Example responses

> 200 Response

```json
null
```

<h3 id="root-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|Inline|

<h3 id="root-responseschema">Response Schema</h3>

<aside class="success">
This operation does not require authentication
</aside>

# Schemas

<h2 id="tocS_Body_predict_api_v1_predict_post">Body_predict_api_v1_predict_post</h2>
<!-- backwards compatibility -->
<a id="schemabody_predict_api_v1_predict_post"></a>
<a id="schema_Body_predict_api_v1_predict_post"></a>
<a id="tocSbody_predict_api_v1_predict_post"></a>
<a id="tocsbody_predict_api_v1_predict_post"></a>

```json
{
  "file": "string"
}

```

Body_predict_api_v1_predict_post

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|file|string|true|none|Imagen del daño vehicular (JPG/PNG)|

<h2 id="tocS_Body_retrain_api_v1_retrain_post">Body_retrain_api_v1_retrain_post</h2>
<!-- backwards compatibility -->
<a id="schemabody_retrain_api_v1_retrain_post"></a>
<a id="schema_Body_retrain_api_v1_retrain_post"></a>
<a id="tocSbody_retrain_api_v1_retrain_post"></a>
<a id="tocsbody_retrain_api_v1_retrain_post"></a>

```json
{
  "k": 2,
  "files": [
    "string"
  ]
}

```

Body_retrain_api_v1_retrain_post

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|k|integer|true|none|Número de clústeres|
|files|[string]|true|none|Imágenes para re-entrenamiento (mínimo K)|

<h2 id="tocS_HTTPValidationError">HTTPValidationError</h2>
<!-- backwards compatibility -->
<a id="schemahttpvalidationerror"></a>
<a id="schema_HTTPValidationError"></a>
<a id="tocShttpvalidationerror"></a>
<a id="tocshttpvalidationerror"></a>

```json
{
  "detail": [
    {
      "loc": [
        "string"
      ],
      "msg": "string",
      "type": "string",
      "input": null,
      "ctx": {}
    }
  ]
}

```

HTTPValidationError

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|detail|[[ValidationError](#schemavalidationerror)]|false|none|none|

<h2 id="tocS_HealthResponse">HealthResponse</h2>
<!-- backwards compatibility -->
<a id="schemahealthresponse"></a>
<a id="schema_HealthResponse"></a>
<a id="tocShealthresponse"></a>
<a id="tocshealthresponse"></a>

```json
{
  "status": "string",
  "model_loaded": true,
  "k_value": 0
}

```

HealthResponse

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|status|string|true|none|none|
|model_loaded|boolean|true|none|none|
|k_value|any|false|none|none|

anyOf

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|integer|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|null|false|none|none|

<h2 id="tocS_HistoryItem">HistoryItem</h2>
<!-- backwards compatibility -->
<a id="schemahistoryitem"></a>
<a id="schema_HistoryItem"></a>
<a id="tocShistoryitem"></a>
<a id="tocshistoryitem"></a>

```json
{
  "id": "string",
  "filename": "string",
  "cluster_id": 0,
  "tipo_dano": "string",
  "severidad": "string",
  "confianza": 0,
  "distancia_centroide": 0,
  "created_at": "string"
}

```

HistoryItem

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|id|string|true|none|none|
|filename|string|true|none|none|
|cluster_id|integer|true|none|none|
|tipo_dano|string|true|none|none|
|severidad|string|true|none|none|
|confianza|number|true|none|none|
|distancia_centroide|number|true|none|none|
|created_at|string|true|none|none|

<h2 id="tocS_HistoryResponse">HistoryResponse</h2>
<!-- backwards compatibility -->
<a id="schemahistoryresponse"></a>
<a id="schema_HistoryResponse"></a>
<a id="tocShistoryresponse"></a>
<a id="tocshistoryresponse"></a>

```json
{
  "data": [
    {
      "id": "string",
      "filename": "string",
      "cluster_id": 0,
      "tipo_dano": "string",
      "severidad": "string",
      "confianza": 0,
      "distancia_centroide": 0,
      "created_at": "string"
    }
  ],
  "total": 0,
  "page": 0,
  "limit": 0
}

```

HistoryResponse

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|data|[[HistoryItem](#schemahistoryitem)]|true|none|none|
|total|integer|true|none|none|
|page|integer|true|none|none|
|limit|integer|true|none|none|

<h2 id="tocS_PredictResponse">PredictResponse</h2>
<!-- backwards compatibility -->
<a id="schemapredictresponse"></a>
<a id="schema_PredictResponse"></a>
<a id="tocSpredictresponse"></a>
<a id="tocspredictresponse"></a>

```json
{
  "id": "string",
  "filename": "string",
  "tipo_dano": "string",
  "severidad": "string",
  "confianza": 0,
  "distancia_centroide": 0,
  "created_at": "string"
}

```

PredictResponse

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|id|string|true|none|none|
|filename|string|true|none|none|
|tipo_dano|string|true|none|none|
|severidad|string|true|none|none|
|confianza|number|true|none|none|
|distancia_centroide|number|true|none|none|
|created_at|string|true|none|none|

<h2 id="tocS_RetrainResponse">RetrainResponse</h2>
<!-- backwards compatibility -->
<a id="schemaretrainresponse"></a>
<a id="schema_RetrainResponse"></a>
<a id="tocSretrainresponse"></a>
<a id="tocsretrainresponse"></a>

```json
{
  "k": 0,
  "silhouette": 0,
  "davies_bouldin": 0,
  "inertia": 0,
  "mapping": [
    {}
  ],
  "trained_at": "string"
}

```

RetrainResponse

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|k|integer|true|none|none|
|silhouette|number|true|none|none|
|davies_bouldin|number|true|none|none|
|inertia|number|true|none|none|
|mapping|[object]|true|none|none|
|trained_at|string|true|none|none|

<h2 id="tocS_ValidationError">ValidationError</h2>
<!-- backwards compatibility -->
<a id="schemavalidationerror"></a>
<a id="schema_ValidationError"></a>
<a id="tocSvalidationerror"></a>
<a id="tocsvalidationerror"></a>

```json
{
  "loc": [
    "string"
  ],
  "msg": "string",
  "type": "string",
  "input": null,
  "ctx": {}
}

```

ValidationError

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|loc|[anyOf]|true|none|none|

anyOf

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|string|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|integer|false|none|none|

continued

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|msg|string|true|none|none|
|type|string|true|none|none|
|input|any|false|none|none|
|ctx|object|false|none|none|

