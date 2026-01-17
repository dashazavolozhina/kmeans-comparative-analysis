"""
ПРОСТОЙ ЭКСПЕРИМЕНТ БЕЗ JUPYTER
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Генерация простых данных
np.random.seed(42)
cluster1 = np.random.randn(100, 2) + [0, 0]
cluster2 = np.random.randn(100, 2) + [5, 5]
cluster3 = np.random.randn(100, 2) + [0, 5]
X = np.vstack([cluster1, cluster2, cluster3])

print("="*60)
print("ПРОСТОЙ ЭКСПЕРИМЕНТ K-MEANS")
print("="*60)

# Импортируем наши алгоритмы
try:
    from src.classic_kmeans import KMeansClassic
    from src.kmeans_plusplus import KMeansPlusPlus
    from src.minibatch_kmeans import MiniBatchKMeans
    print("✅ Алгоритмы успешно импортированы")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Убедитесь, что файлы в папке src/ существуют")
    exit()

# Тестируем классический K-means
print("\n1. Тестируем Classic K-means:")
kmeans = KMeansClassic(n_clusters=3, random_state=42)
start = time.time()
kmeans.fit(X)
end = time.time()
print(f"   Время: {end-start:.3f}с")
print(f"   Инерция: {kmeans.inertia_:.2f}")

# Сохраняем график
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
            c='red', marker='X', s=200, label='Centroids')
plt.title("Classic K-means Results")
plt.legend()
plt.savefig('results/plots/simple_result.png')
print("✅ График сохранен: results/plots/simple_result.png")
plt.show()