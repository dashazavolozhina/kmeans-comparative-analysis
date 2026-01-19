import numpy as np
import matplotlib.pyplot as plt
from src.classic_kmeans import KMeansClassic

# Создаем простые тестовые данные
np.random.seed(42)
# Три кластера в 2D пространстве
cluster1 = np.random.randn(50, 2) + np.array([0, 0])
cluster2 = np.random.randn(50, 2) + np.array([5, 5])
cluster3 = np.random.randn(50, 2) + np.array([-5, 5])

X = np.vstack([cluster1, cluster2, cluster3])

print("Форма данных:", X.shape)
print("Первые 5 точек:")
print(X[:5])

# Тестируем классический K-means
print("\n--- Тестируем Classic K-means ---")
kmeans = KMeansClassic(n_clusters=3, max_iter=50, random_state=42)
kmeans.fit(X)

print(f"Количество итераций: {kmeans.n_iter_}")
print(f"Инерция: {kmeans.inertia_:.4f}")
print("Центроиды:")
print(kmeans.centroids)

# Визуализация
plt.figure(figsize=(10, 8))

# Исходные данные
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title("Исходные данные")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Результаты кластеризации
plt.subplot(2, 2, 2)
colors = ['red', 'green', 'blue']
for i in range(3):
    cluster_points = X[kmeans.labels_ == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                color=colors[i], alpha=0.6, label=f'Cluster {i}')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
            color='black', marker='X', s=200, label='Centroids')
plt.title("Результаты K-means")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.tight_layout()
plt.savefig('results/plots/first_test.png')
print("\nГрафик сохранен в results/plots/first_test.png")