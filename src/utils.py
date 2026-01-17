import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
import time
import json
import os

def generate_synthetic_datasets():
    """
    Генерация трех типов синтетических данных для экспериментов
    
    Возвращает:
    -----------
    datasets : dict
        Словарь с тремя датасетами:
        - 'blobs_clear': четкие кластеры
        - 'blobs_overlap': перекрывающиеся кластеры  
        - 'moons': нелинейно разделимые данные
    """
    np.random.seed(42)
    
    # 1. Четкие кластеры
    X_clear, y_clear = make_blobs(
        n_samples=1000, 
        centers=4, 
        cluster_std=0.6, 
        random_state=42
    )
    
    # 2. Перекрывающиеся кластеры
    X_overlap, y_overlap = make_blobs(
        n_samples=1000,
        centers=4,
        cluster_std=1.5,  # Больше разброс = больше перекрытие
        random_state=42
    )
    
    # 3. Данные в форме лун (сложнее для K-means)
    X_moons, y_moons = make_moons(
        n_samples=1000,
        noise=0.1,
        random_state=42
    )
    
    datasets = {
        'blobs_clear': (X_clear, y_clear, 'Четкие кластеры (blobs)'),
        'blobs_overlap': (X_overlap, y_overlap, 'Перекрывающиеся кластеры'),
        'moons': (X_moons, y_moons, 'Данные "луны" (нелинейные)')
    }
    
    return datasets

def compute_metrics(X, labels, centroids):
    """
    Вычисление различных метрик качества кластеризации
    
    Возвращает:
    -----------
    metrics : dict
        Словарь с вычисленными метриками
    """
    metrics = {}
    
    # 1. Inertia (within-cluster sum of squares)
    inertia = 0
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            inertia += np.sum((cluster_points - centroids[i]) ** 2)
    metrics['inertia'] = inertia
    
    # 2. Размеры кластеров
    cluster_sizes = [np.sum(labels == i) for i in range(len(centroids))]
    metrics['cluster_sizes'] = cluster_sizes
    metrics['size_std'] = np.std(cluster_sizes)  # стандартное отклонение размеров
    
    # 3. Среднее расстояние до центроида
    avg_distances = []
    for i in range(len(centroids)):
        if cluster_sizes[i] > 0:
            distances = np.sqrt(np.sum((X[labels == i] - centroids[i]) ** 2, axis=1))
            avg_distances.append(np.mean(distances))
    metrics['avg_distance_to_centroid'] = np.mean(avg_distances)
    
    return metrics

def plot_clusters(X, labels, centroids, title, save_path=None):
    """
    Визуализация результатов кластеризации
    
    Параметры:
    ----------
    X : array, данные
    labels : array, метки кластеров
    centroids : array, центроиды
    title : str, заголовок графика
    save_path : str, путь для сохранения (опционально)
    """
    plt.figure(figsize=(10, 8))
    
    # Цвета для кластеров
    colors = plt.cm.tab10(np.linspace(0, 1, len(centroids)))
    
    # Рисуем точки
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       color=colors[i], alpha=0.6, s=30, label=f'Cluster {i}')
    
    # Рисуем центроиды
    plt.scatter(centroids[:, 0], centroids[:, 1], 
                color='black', marker='X', s=200, label='Centroids', linewidths=2)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        # Создаем папку если нет
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

def plot_convergence(inertia_history, title, save_path=None):
    """
    Визуализация сходимости алгоритма (изменение инерции)
    
    Параметры:
    ----------
    inertia_history : list, история изменения инерции
    title : str, заголовок графика
    save_path : str, путь для сохранения (опционально)
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(inertia_history, marker='o', markersize=5, linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel('Итерация', fontsize=12)
    plt.ylabel('Инерция', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

def save_results(results, filename):
    """
    Сохранение результатов экспериментов в JSON файл
    
    Параметры:
    ----------
    results : dict, результаты экспериментов
    filename : str, имя файла для сохранения
    """
    # Конвертируем numpy типы в стандартные Python типы
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    # Создаем папку если нет
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"Результаты сохранены в {filename}")

def load_results(filename):
    """
    Загрузка результатов из JSON файла
    """
    with open(filename, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results

def compare_algorithms(algorithms, X, n_runs=10):
    """
    Сравнение алгоритмов с многократными запусками
    
    Параметры:
    ----------
    algorithms : list of tuples, [(name, model), ...]
    X : array, данные
    n_runs : int, количество запусков для оценки стабильности
    
    Возвращает:
    -----------
    comparison_results : dict, результаты сравнения
    """
    comparison_results = {}
    
    for name, model_class in algorithms:
        print(f"\nТестируем {name}...")
        
        inertias = []
        times = []
        n_iters_list = []
        
        for run in range(n_runs):
            # Создаем новый экземпляр модели для каждого запуска
            model = model_class
            
            # Измеряем время
            start_time = time.time()
            model.fit(X)
            end_time = time.time()
            
            inertias.append(model.inertia_)
            times.append(end_time - start_time)
            n_iters_list.append(model.n_iter_)
        
        # Собираем статистику
        comparison_results[name] = {
            'inertia_mean': np.mean(inertias),
            'inertia_std': np.std(inertias),
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'n_iter_mean': np.mean(n_iters_list),
            'all_inertias': inertias,
            'all_times': times
        }
        
        print(f"  Средняя инерция: {np.mean(inertias):.2f} ± {np.std(inertias):.2f}")
        print(f"  Среднее время: {np.mean(times):.3f}с ± {np.std(times):.3f}с")
    
    return comparison_results