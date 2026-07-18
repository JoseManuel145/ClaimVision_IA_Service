# Estándares de Captura para Documentos

## Credencial INE (Instituto Nacional Electoral)

### Especificaciones Físicas

| Característica | Valor |
|----------------|-------|
| Dimensiones físicas | 85.6mm × 54mm (tarjeta estándar ISO/IEC 7810 ID-1) |
| Proporción de aspecto | Landscape: 1.585:1 / Portrait: 0.631:1 |
| Resolución mínima | 800 × 500 píxeles |
| Resolución recomendada | 1200 × 800 píxeles o mayor |

### Orientaciones Aceptadas

- **Landscape (horizontal):** Proporción ~1.59:1 ← *Recomendada*
- **Portrait (vertical):** Proporción ~0.63:1

### Requisitos de Calidad

| Criterio | Mínimo | Recomendado |
|----------|--------|-------------|
| Nitidez (varianza Laplaciana) | > 50 | > 100 |
| Brillo promedio | 50-200 (escala 0-255) | 80-170 |
| Contraste | Visible | Alto |
| Ruido | Bajo | Mínimo |

### Errores Comunes que Rechazan la Imagen

1. **Resolución insuficiente:** Imagen menor a 800×500 píxeles
2. **Imagen borrosa:** Nitidez < 50 (movimiento de cámara, enfoque incorrecto)
3. **Imagen oscura:** Brillo promedio < 50 (poca luz, sombras)
4. **Imagen sobreexpuesta:** Brillo promedio > 200 (destello, luz directa)
5. **Aspecto incorrecto:** Proporción no coincide con ~0.63 o ~1.59
6. **Recorte incorrecto:** La INE no está completamente visible

### Ejemplo de Captura Correcta

```
┌─────────────────────────────────┐
│                                 │
│   ┌─────────────────────────┐   │
│   │  FRENTE DE LA INE       │   │
│   │  - Nombre visible       │   │
│   │  - CURP visible         │   │
│   │  - Foto visible         │   │
│   │  - Sexo visible         │   │
│   └─────────────────────────┘   │
│                                 │
└─────────────────────────────────┘
```

### Mensajes de Error para el Frontend

Cuando la validación falla, el backend retorna:

```json
{
  "error": "La imagen de la INE no cumple con los estándares de calidad",
  "details": [
    "Resolución insuficiente: 640x480 (mínimo 800x500)",
    "Imagen borrosa (nitidez: 23, mínima: 50)",
    "Aspecto incorrecto: 1.33 (esperado ~0.63 o ~1.59)"
  ],
  "suggestion": "Capture la INE con buena iluminación, sin movimiento y en posición horizontal"
}
```

### Acciones del Frontend

1. **MostrarToast/Error:** Desplegar mensaje de error específico
2. **Solicitar reintento:** Pedir al usuario que tome otra foto
3. **Guía visual:** Mostrar ejemplo de cómo debe verse la foto
4. **Pre-validación opcional:** Verificar dimensiones antes de enviar

---

## Póliza de Seguro

### Especificaciones Físicas

| Característica | Valor |
|----------------|-------|
| Dimensiones físicas | Carta: 215.9mm × 279.4mm (8.5" × 11") |
| Proporción de aspecto | Portrait: 0.773:1 / Landscape: 1.294:1 |
| Resolución mínima | 1000 × 700 píxeles |
| Resolución recomendada | 1500 × 1000 píxeles o mayor |

### Orientaciones Aceptadas

- **Portrait (vertical):** Proporción ~0.77:1 ← *Recomendada*
- **Landscape (horizontal):** Proporción ~1.29:1

### Requisitos de Calidad

| Criterio | Mínimo | Recomendado |
|----------|--------|-------------|
| Nitidez (varianza Laplaciana) | > 30 | > 80 |
| Brillo promedio | 60-200 (escala 0-255) | 90-170 |
| Contraste | Visible | Alto |
| Bordes visibles | Sí | Sí |

### Errores Comunes que Rechazan el Documento

1. **Resolución insuficiente:** Imagen menor a 1000×700 píxeles
2. **Documento borrosa:** Nitidez < 30
3. **Documento oscura:** Brillo promedio < 60
4. **Documento sobreexpuesta:** Brillo promedio > 200
5. **Aspecto incorrecto:** Proporción no coincide con ~0.77 o ~1.29
6. **Página incompleta:** Bordes del documento no visibles

### Ejemplo de Captura Correcta

```
┌─────────────────────────────────┐
│                                 │
│   ┌─────────────────────────┐   │
│   │  PÓLIZA DE SEGURO       │   │
│   │  - Número de póliza     │   │
│   │  - Aseguradora          │   │
│   │  - Datos del vehículo   │   │
│   │  - Vigencia             │   │
│   └─────────────────────────┘   │
│                                 │
└─────────────────────────────────┘
```

### Mensajes de Error para el Frontend

```json
{
  "error": "La imagen de la póliza no cumple con los estándares de calidad",
  "details": [
    "Resolución insuficiente: 640x480 (mínimo 1000x700)",
    "Documento borrosa (nitidez: 15, mínima: 30)"
  ],
  "suggestion": "Capture la póliza completa con buena iluminación y bordes visibles"
}
```

---

## Código de Respuesta HTTP

| Código | Significado |
|--------|-------------|
| 200 | Éxito - Extracción completada |
| 400 | Error de validación - Imagen no cumple estándares |
| 413 | Archivo demasiado grande (>10MB) |
| 415 | Tipo de archivo no soportado |
| 502 | Error de comunicación con servicio de IA |

---

## Implementación Técnica (Backend)

### Validación de Imagen

```python
# En app/modules/ocr/infra/validation/image_validator.py
from app.modules.ocr.infra.validation.image_validator import ImageValidator

validator = ImageValidator()

# Para INE
result = await validator.validate_ine_image(image_bytes)
if not result.is_valid:
    raise HTTPException(
        status_code=400,
        detail={
            "error": "La imagen de la INE no cumple estándares",
            "details": result.errors,
            "suggestion": "Capture la INE con buena iluminación..."
        }
    )

# Para Póliza
result = await validator.validate_poliza_image(image_bytes)
```

### Respuesta de Error Estructurada

```json
{
  "error": "string (mensaje principal)",
  "details": ["string array (errores específicos)"],
  "suggestion": "string (cómo corregir)"
}
```
