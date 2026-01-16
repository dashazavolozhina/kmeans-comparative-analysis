import numpy as np
import time
import matplotlib.pyplot as plt
from src.classic_kmeans import KMeansClassic
from src.kmeans_plusplus import KMeansPlusPlus
from src.minibatch_kmeans import MiniBatchKMeans

# Генерируем тестовые данные
np.random.seed(42)
n_samples = 1000
n_clusters = 3

# Три хорошо разделенных кластера
cluster1 = np.random.randn(n_samples, 2) * 0.5 + np.array([0, 0])
cluster2 = np.random.randn(n_samples, 2) * 0.5 + np.array([5, 5])
cluster3 = np.random.randn(n_samples, 2) * 0.5 + np.array([-5, 5])

X = np.vstack([cluster1, cluster2, cluster3])

print("=" * 60)
print("СРАВНЕНИЕ АЛГОРИТМОВ K-MEANS")
print("=" * 60)
print(f"Количество точек: {X.shape[0]}")
print(f"Количество кластеров: {n_clusters}")
print()

# Список алгоритмов для тестирования
algorithms = [
    ("Classic K-means", KMeansClassic(n_clusters=n_clusters, random_state=42)),
    ("K-means++", KMeansPlusPlus(n_clusters=n_clusters, random_state=42)),
    ("Mini-Batch K-means", MiniBatchKMeans(n_clusters=n_clusters, batch_size=100, random_state=42))
]

results = []

for name, model in algorithms:
    print(f"\n--- {name} ---")
    
    # Измеряем время выполнения
    start_time = time.time()
    model.fit(X)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    print(f"Время выполнения: {execution_time:.4f} сек")
    print(f"Количество итераций: {model.n_iter_}")
    print(f"Инерция: {model.inertia_:.4f}")
    
    results.append({
        'name': name,
        'time': execution_time,
        'inertia': model.inertia_,
        'iterations': model.n_iter_,
        'model': model
    })

# Визуализация результатов
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Исходные данные
axes[0, 0].scatter(X[:, 0], X[:, 1], alpha=0.3, s=10)
axes[0, 0].set_title("Исходные данные")
axes[0, 0].set_xlabel("Feature 1")
axes[0, 0].set_ylabel("Feature 2")

# Результаты для каждого алгоритма
colors = ['red', 'green', 'blue']
for idx, (result, color) in enumerate(zip(results, colors)):
    row = (idx + 1) // 2
    col = (idx + 1) % 2
    
    model = result['model']
    labels = model.labels_
    
    for cluster_id in range(n_clusters):
        cluster_points = X[labels == cluster_id]
        axes[row, col].scatter(cluster_points[:, 0], cluster_points[:, 1], 
                              color=color, alpha=0.3, s=10)
    
    # Центроиды
    axes[row, col].scatter(model.centroids[:, 0], model.centroids[:, 1], 
                          color='black', marker='X', s=200, label='Centroids')
    
    axes[row, col].set_title(f"{result['name']}\n"
                           f"Время: {result['time']:.3f}с, Инерция: {result['inertia']:.2f}")
    axes[row, col].set_xlabel("Feature 1")
    axes[row, col].set_ylabel("Feature 2")

plt.tight_layout()
plt.savefig('results/plots/comparison_results.png')
print(f"\nГрафик сохранен в results/plots/comparison_results.png")

# Сводная таблица результатов
print("\n" + "=" * 60)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 60)
print(f"{'Алгоритм':<20} {'Время (с)':<12} {'Инерция':<12} {'Итерации':<10}")
print("-" * 60)

for result in results:
    print(f"{result['name']:<20} {result['time']:<12.4f} "
          f"{result['inertia']:<12.2f} {result['iterations']:<10}")