import json
import httpx
from groq import Groq
from app.modules.nlp.domain.models import DamageEntity

_SYS = (
    "Eres un analizador de danos vehiculares. "
    'Dado un texto, responde JSON con clave "d" = array de danos. '
    'Cada dano: "t"=tipo (perdida_potencia, ruidos_anormales, pedal_esponjoso, etc), '
    '"s"=severidad (Alto/Medio/Bajo), "p"=parte (motor/frenos/carroceria/etc), '
    '"x"=sintoma (texto exacto), "c"=confianza (0-1). '
    'Sin danos: {"d":[]}'
)


class GroqExtractor:
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        ssl_verify: bool = True,
    ):
        http_client = httpx.Client(verify=ssl_verify, timeout=60.0)
        self.client = Groq(api_key=api_key, http_client=http_client)
        self.model = model

    async def extraer_danos(self, texto: str) -> list[DamageEntity]:
        for _ in range(2):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYS},
                        {"role": "user", "content": texto},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=256,
                )
                raw = resp.choices[0].message.content or '{"d":[]}'
                parsed = json.loads(raw)
                items = parsed.get("d", []) if isinstance(parsed, dict) else []
                return [
                    DamageEntity(
                        tipo_dano=i.get("t", ""),
                        severidad=i.get("s", "Medio"),
                        parte_afectada=i.get("p", ""),
                        sintoma=i.get("x", ""),
                        confianza=float(i.get("c", 0)),
                    )
                    for i in items
                    if isinstance(i, dict)
                ]
            except (json.JSONDecodeError, Exception):
                continue
        return []
