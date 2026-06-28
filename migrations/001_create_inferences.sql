CREATE TABLE IF NOT EXISTS inferences (
    id              VARCHAR       PRIMARY KEY,
    filename        VARCHAR(255)  NOT NULL,
    cluster_id      INTEGER       NOT NULL,
    tipo_dano       VARCHAR(50)   NOT NULL,
    severidad       VARCHAR(10)   NOT NULL,
    confianza       FLOAT         NOT NULL,
    distancia_centroide FLOAT     NOT NULL,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inferences_created_at ON inferences (created_at DESC);
