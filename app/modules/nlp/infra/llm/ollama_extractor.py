import json
import httpx
from app.modules.nlp.domain.models import DamageEntity

DAMAGE_CATALOG = """
Motor: perdida de potencia, sobrecalentamiento, humo excesivo (blanco/azul/negro), consumo excesivo de aceite, dificultad al arrancar, marcha minima irregular, tirones, ruidos anormales (golpeteo, silbido, rozamiento)
Transmision: dificil cambiar marchas, patina el embrague, ruidos al cambiar, vibraciones en palanca, perdida de fuerza en subidas
Suspension/Direccion: desviacion del vehiculo, vibraciones en volante, ruidos al pasar baches, direccion dura o floja
Frenos: chillido al frenar, pedal esponjoso/duro, vibraciones al frenar, desviacion al frenar, perdida de eficacia
Electrico: luces parpadean, bateria se descarga rapido, fallo de sensores/testigos, arranque debil
Escape: ruido excesivo, olor a combustible, perdida de potencia por restriccion
Carroceria: filtraciones de agua, puertas no cierran bien, ruidos de torsion, paneles desalineados
"""


class OllamaExtractor:
    def __init__(
        self,
        base_url: str = "http://ollama:11434",
        model: str = "qwen2.5:3b",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def extraer_danos(self, texto: str) -> list[DamageEntity]:
        prompt = f"""Dado el siguiente texto de un conductor describiendo danos en su vehiculo, extrae los danos NO VISIBLES mencionados. Responde SOLO con un JSON array.

Texto: "{texto}"

Cada elemento del array debe tener:
- tipo_dano: string (ej: "perdida_potencia", "golpe_trasero")
- severidad: "Alto" | "Medio" | "Bajo"
- parte_afectada: string (ej: "motor", "carroceria")
- sintoma: string (texto exacto del sintoma)
- confianza: float entre 0 y 1

Catalogo de referencia (usar estos valores de tipo_dano cuando corresponda):
{DAMAGE_CATALOG}

Si no hay danos detectables, responde: []"""

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data.get("response", "[]")

            parsed = json.loads(raw)
        except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError):
            return []

        if isinstance(parsed, list):
            pass
        elif isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break
            else:
                parsed = [parsed]
        else:
            return []

        entidades = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            entidades.append(
                DamageEntity(
                    tipo_dano=item.get("tipo_dano", ""),
                    severidad=item.get("severidad", "Medio"),
                    parte_afectada=item.get("parte_afectada", ""),
                    sintoma=item.get("sintoma", ""),
                    confianza=float(item.get("confianza", 0)),
                )
            )
        return entidades
