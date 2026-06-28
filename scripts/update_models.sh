#!/bin/bash
# Updates model files from EDA training output
# Run this after training completes

EDA_MODELS="../Mineria de Datos/EDA/models"
TARGET="models"

echo "Updating models from $EDA_MODELS ..."

cp "$EDA_MODELS"/encoder_best.pth "$TARGET"/encoder_best.pth
cp "$EDA_MODELS"/encoder_config.json "$TARGET"/encoder_config.json
cp "$EDA_MODELS"/kmeans.pkl "$TARGET"/kmeans.pkl

# Convert centroids to float64 for scikit-learn compatibility
python3 -c "
import pickle, numpy as np, json
with open('$TARGET/kmeans.pkl', 'rb') as f:
    km = pickle.load(f)
km.cluster_centers_ = km.cluster_centers_.astype(np.float64)
with open('$TARGET/kmeans.pkl', 'wb') as f:
    pickle.dump(km, f)
print('KMeans centroids converted to float64')
"

# Regenerate cluster mapping
python3 -c "
import pandas as pd, numpy as np, json, pickle

with open('$TARGET/kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)
X = np.load('$EDA_MODELS/latent_vectors.npy')
df = pd.read_csv('$EDA_MODELS/latent_metadata.csv')
labels = kmeans.predict(X)

label_names = {1: 'Crack', 2: 'Scratch', 3: 'Flat tire', 4: 'Dent', 5: 'Glass shatter', 6: 'Lamp broken'}
damage_types = ['Fractura', 'Rayadura', 'Deformacion', 'Abolladura', 'Rotura_Cristal', 'Desperfecto_Multiple']

clusters = []
for c in range(kmeans.n_clusters):
    mask = labels == c
    subset = df[mask]
    dominant_label = int(subset['label'].value_counts().idxmax())
    dominant_pct = round(float(subset['label'].value_counts(normalize=True).max() * 100), 1)
    center = kmeans.cluster_centers_[c]
    distances = np.linalg.norm(X[mask] - center, axis=1)
    
    idx = c % len(damage_types)
    tipo = damage_types[idx]
    
    clusters.append({
        'id': int(c),
        'tipo_dano': tipo,
        'severidad_base': 'Medio',
        'count': int(mask.sum()),
        'dominant_label': dominant_label,
        'dominant_label_name': label_names.get(dominant_label, 'Desconocido'),
        'dominant_pct': dominant_pct,
        'avg_distance': round(float(distances.mean()), 4),
        'max_distance': round(float(distances.max()), 4),
    })

mapping = {
    'version': 1,
    'k': kmeans.n_clusters,
    'silhouette': 0.0,
    'trained_at': '2026-06-27',
    'label_names': {str(k): v for k, v in label_names.items()},
    'clusters': clusters,
}

with open('$TARGET/cluster_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)
print('cluster_mapping.json regenerated')
"

echo "Models updated successfully!"
