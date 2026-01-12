# Кастеризация данных
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

OUT_DIR = "clustering_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. ЗАГРУЗКА ДАННЫХ
wine = load_wine()
df = pd.DataFrame(wine.data, columns=[c.replace('/', '_').replace(' ', '_') for c in wine.feature_names])
print(f"Загружен датасет: {df.shape}")
print(f"Количество признаков: {df.shape[1]}")
print(f"Количество объектов: {df.shape[0]}")
print(f"\nНазвания признаков: {list(df.columns)}")

# 2. ПРОВЕРКА ДАННЫХ И СТАНДАРТИЗАЦИЯ
print("\n2. ПРОВЕРКА ДАННЫХ И СТАНДАРТИЗАЦИЯ")
print("-" * 40)

print("Проверка пропусков в данных:")
print(df.isna().sum())

print("\nОписательная статистика до стандартизации:")
print(df.describe().round(3))

# Стандартизация данных - важный шаг для кластеризации!
# Приводим все признаки к одинаковому масштабу (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
print(f"\nДанные стандартизированы. Размерность: {X_scaled.shape}")

# 3. PCA ДЛЯ ВИЗУАЛИЗАЦИИ
print("\n3. PCA ДЛЯ ВИЗУАЛИЗАЦИИ")
print("-" * 30)

# PCA (Principal Component Analysis) - метод уменьшения размерности
# Позволяет визуализировать многомерные данные в 2D/3D
pca = PCA(n_components=2, random_state=42)
X_pca2 = pca.fit_transform(X_scaled)

print(f"Объясненная дисперсия компонент: {pca.explained_variance_ratio_}")
print(f"Суммарная объясненная дисперсия: {sum(pca.explained_variance_ratio_):.3f}")

# Визуализация исходных данных после PCA
plt.figure(figsize=(8, 6))
plt.scatter(X_pca2[:, 0], X_pca2[:, 1], s=30, alpha=0.7)
plt.title("PCA (2 компоненты) — исходные данные")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} дисперсии)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} дисперсии)")
plt.grid(True, alpha=0.3)
plt.savefig(f"{OUT_DIR}/pca_raw.png", bbox_inches='tight', dpi=150)
plt.close()
print("Сохранен график PCA исходных данных")

# 4. KMEANS: ПОДБОР ОПТИМАЛЬНОГО КОЛИЧЕСТВА КЛАСТЕРОВ
print("\n4. ПОДБОР ОПТИМАЛЬНОГО K ДЛЯ K-MEANS")
print("-" * 45)

# Тестируем разные значения k от 2 до 8
ks = range(2, 9)
inertias, silhouettes, db_scores = [], [], []

print("k\tInertia\t\tSilhouette\tDavies-Bouldin") # Инертность = ΣΣ ||x - μ||² Сил - s(i) = (b(i) - a(i)) / max(a(i), b(i))
print("-" * 50) # Индекс Дэвиса-Боулдина - DB = (1/k) * Σ max[(σ_i + σ_j) / d(c_i, c_j)]

for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

    print(f"{k}\t{km.inertia_:.2f}\t\t{silhouettes[-1]:.4f}\t\t{db_scores[-1]:.4f}")

# Сохраняем результаты в DataFrame
results = pd.DataFrame({
    'k': ks,
    'inertia': inertias,
    'silhouette': silhouettes,
    'davies_bouldin': db_scores
})

# Визуализация метода локтя
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(ks, inertias, marker='o', linewidth=2, markersize=8)
plt.title("Метод локтя: Inertia vs k")
plt.xlabel("Количество кластеров (k)")
plt.ylabel("Inertia (внутрикластерная дисперсия)")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(ks, silhouettes, marker='s', color='green', linewidth=2, markersize=8)
plt.title("Silhouette Score vs k")
plt.xlabel("Количество кластеров (k)")
plt.ylabel("Silhouette Score")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/k_selection.png", bbox_inches='tight', dpi=150)
plt.close()

# Выбираем лучшее k по максимальному silhouette score
best_k = int(results.loc[results['silhouette'].idxmax(), 'k'])
best_silhouette = results.loc[results['silhouette'].idxmax(), 'silhouette']

print(f"\nЛучшее k по silhouette score: {best_k} (score = {best_silhouette:.4f})")

# 5. KMEANS С ОПТИМАЛЬНЫМ K
print("\n5. K-MEANS С ОПТИМАЛЬНЫМ K")
print("-" * 30)

kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=20)
labels_km = kmeans_best.fit_predict(X_scaled)

# Визуализация результатов K-means
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=labels_km, s=30, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Кластер')
plt.title(f"K-Means кластеризация (k={best_k})")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} дисперсии)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} дисперсии)")
plt.grid(True, alpha=0.3)
plt.savefig(f"{OUT_DIR}/kmeans_k{best_k}.png", bbox_inches='tight', dpi=150)
plt.close()

print(f"K-Means завершен. Создано {best_k} кластеров")

# 6. ИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ
print("\n6. ИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ")
print("-" * 35)

# Agglomerative Clustering - восходящий иерархический подход
agg = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
labels_agg = agg.fit_predict(X_scaled)

# Визуализация иерархической кластеризации
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=labels_agg, s=30, cmap='plasma', alpha=0.7)
plt.colorbar(scatter, label='Кластер')
plt.title(f"Иерархическая кластеризация (k={best_k})")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} дисперсии)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} дисперсии)")
plt.grid(True, alpha=0.3)
plt.savefig(f"{OUT_DIR}/agglomerative_k{best_k}.png", bbox_inches='tight', dpi=150)
plt.close()

# Метрики качества для иерархической кластеризации
silhouette_agg = silhouette_score(X_scaled, labels_agg)
db_agg = davies_bouldin_score(X_scaled, labels_agg)

print("Метрики иерархической кластеризации:")
print(f"Silhouette Score = {silhouette_agg:.4f}")
print(f"Davies-Bouldin Index = {db_agg:.4f}")

# 7. DBSCAN - КЛАСТЕРИЗАЦИЯ ПО ПЛОТНОСТИ
print("\n7. DBSCAN - КЛАСТЕРИЗАЦИЯ ПО ПЛОТНОСТИ")
print("-" * 45)

# Подбор параметров для DBSCAN
eps_values = [0.3, 0.5, 0.7, 0.9, 1.1]
min_samples_values = [3, 5, 7]
dbscan_results = []

print("Подбор параметров DBSCAN:")
print("eps\tmin_samples\tclusters\tnoise\t\tsilhouette")
print("-" * 65)

for eps in eps_values:
    for ms in min_samples_values:
        db = DBSCAN(eps=eps, min_samples=ms)
        labels_db = db.fit_predict(X_scaled)

        # Количество кластеров (исключая шум -1)
        n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
        n_noise = list(labels_db).count(-1)

        if n_clusters > 0:
            sil = silhouette_score(X_scaled, labels_db)
            dbs = davies_bouldin_score(X_scaled, labels_db)
        else:
            sil, dbs = np.nan, np.nan

        dbscan_results.append({
            'eps': eps, 'min_samples': ms, 'n_clusters': n_clusters,
            'n_noise': n_noise, 'silhouette': sil, 'davies_bouldin': dbs
        })

        if not np.isnan(sil):
            sil_str = f"{sil:.4f}"
        else:
            sil_str = "N/A"
        print(f"{eps}\t{ms}\t\t{n_clusters}\t\t{n_noise}\t\t{sil_str}")

df_dbscan = pd.DataFrame(dbscan_results)

# Выбираем лучшие параметры DBSCAN
valid_db = df_dbscan.dropna(subset=['silhouette'])
if not valid_db.empty:
    best_db = valid_db.loc[valid_db['silhouette'].idxmax()]
    print(f"\nЛучшие параметры DBSCAN:")
    print(f"eps = {best_db['eps']}, min_samples = {best_db['min_samples']}")
    print(f"Кластеров: {best_db['n_clusters']}, Шум: {best_db['n_noise']}")
    print(f"Silhouette: {best_db['silhouette']:.4f}")

    # Визуализация лучшего DBSCAN
    db_best = DBSCAN(eps=float(best_db['eps']), min_samples=int(best_db['min_samples']))
    labels_db = db_best.fit_predict(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=labels_db, s=30, cmap='cool', alpha=0.7)
    plt.colorbar(scatter, label='Кластер')
    plt.title(f"DBSCAN (eps={best_db['eps']}, min_samples={best_db['min_samples']})")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} дисперсии)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} дисперсии)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUT_DIR}/dbscan_best.png", bbox_inches='tight', dpi=150)
    plt.close()
else:
    print("\nDBSCAN не смог выделить валидные кластеры для данных параметров.")

# 8. СРАВНИТЕЛЬНАЯ ОЦЕНКА АЛГОРИТМОВ
print("\n8. СРАВНИТЕЛЬНАЯ ОЦЕНКА АЛГОРИТМОВ")
print("-" * 40)

print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
print("Алгоритм\t\tКластеров\tSilhouette\tDavies-Bouldin")
print("-" * 65)

# K-Means метрики
silhouette_km = silhouette_score(X_scaled, labels_km)
db_km = davies_bouldin_score(X_scaled, labels_km)
print(f"K-Means\t\t{best_k}\t\t{silhouette_km:.4f}\t\t{db_km:.4f}")

# Agglomerative метрики
print(f"Agglomerative\t{best_k}\t\t{silhouette_agg:.4f}\t\t{db_agg:.4f}")

# DBSCAN метрики (если есть валидные результаты)
if not valid_db.empty:
    print(f"DBSCAN\t\t{best_db['n_clusters']}\t\t{best_db['silhouette']:.4f}\t\t{best_db['davies_bouldin']:.4f}")

# 9. АНАЛИЗ КЛАСТЕРОВ K-MEANS
print("\n9. АНАЛИЗ КЛАСТЕРОВ K-MEANS")
print("-" * 35)

# Добавляем метки кластеров к исходным данным
df['cluster_km'] = labels_km

# Группируем по кластерам и вычисляем средние значения
cluster_means = df.groupby('cluster_km').mean().round(3)
cluster_sizes = df['cluster_km'].value_counts().sort_index()

print("Размеры кластеров:")
for cluster, size in cluster_sizes.items():
    print(f"Кластер {cluster}: {size} объектов ({size / len(df) * 100:.1f}%)")

print(f"\nСредние значения признаков по кластерам:")
print(cluster_means)

# Сохраняем результаты
cluster_means.to_csv(f"{OUT_DIR}/cluster_means.csv")
results.to_csv(f"{OUT_DIR}/k_selection_results.csv", index=False)
df_dbscan.to_csv(f"{OUT_DIR}/dbscan_results.csv", index=False)

# 10. ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ АЛГОРИТМОВ
print("\n10. СОЗДАНИЕ СРАВНИТЕЛЬНЫХ ГРАФИКОВ")
print("-" * 45)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Исходные данные
axes[0, 0].scatter(X_pca2[:, 0], X_pca2[:, 1], s=30, alpha=0.7)
axes[0, 0].set_title("Исходные данные (PCA)")
axes[0, 0].set_xlabel("PC1")
axes[0, 0].set_ylabel("PC2")
axes[0, 0].grid(True, alpha=0.3)

# K-Means
scatter1 = axes[0, 1].scatter(X_pca2[:, 0], X_pca2[:, 1], c=labels_km, s=30, cmap='viridis', alpha=0.7)
axes[0, 1].set_title(f"K-Means (k={best_k})")
axes[0, 1].set_xlabel("PC1")
axes[0, 1].set_ylabel("PC2")
axes[0, 1].grid(True, alpha=0.3)

# Agglomerative
scatter2 = axes[1, 0].scatter(X_pca2[:, 0], X_pca2[:, 1], c=labels_agg, s=30, cmap='plasma', alpha=0.7)
axes[1, 0].set_title(f"Agglomerative (k={best_k})")
axes[1, 0].set_xlabel("PC1")
axes[1, 0].set_ylabel("PC2")
axes[1, 0].grid(True, alpha=0.3)

# DBSCAN (если есть)
if not valid_db.empty:
    scatter3 = axes[1, 1].scatter(X_pca2[:, 0], X_pca2[:, 1], c=labels_db, s=30, cmap='cool', alpha=0.7)
    axes[1, 1].set_title(f"DBSCAN (clusters={best_db['n_clusters']})")
else:
    axes[1, 1].text(0.5, 0.5, 'DBSCAN\nне нашел\nкластеры',
                    ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_title("DBSCAN")
axes[1, 1].set_xlabel("PC1")
axes[1, 1].set_ylabel("PC2")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/comparison_all_methods.png", bbox_inches='tight', dpi=150)
plt.close()

print(f"Все результаты сохранены в папке: {OUT_DIR}")
print("\nСозданные файлы:")
for file in os.listdir(OUT_DIR):
    print(f"  - {file}")