import pandas as pd
import numpy as np

# Caricamento dei risultati calcolati
df = pd.read_csv('stl_reconstruction_metrics.csv')

def get_stats(series):
    return {
        '1quart': series.quantile(0.25),
        'median': series.median(),
        '3quart': series.quantile(0.75),
        '99perc': series.quantile(0.99)
    }

# Calcolo metriche per Distanza Euclidea
dist_stats = get_stats(df['euclidean_distance'])

# Calcolo metriche per Cosine Similarity
cos_stats = get_stats(df['cosine_similarity'])

# Formattazione Tabella (Stile Paper)
data = {
    "Metric": ["Euclidean Dist d(p, p^)", "Cosine Sim cos(p, p^)"],
    "1quart": [dist_stats['1quart'], cos_stats['1quart']],
    "median": [dist_stats['median'], cos_stats['median']],
    "3quart": [dist_stats['3quart'], cos_stats['3quart']],
    "99perc": [dist_stats['99perc'], cos_stats['99perc']]
}

stats_df = pd.DataFrame(data)

print("\n### Comparison of Results (Your Checkpoint) ###")
print(stats_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

