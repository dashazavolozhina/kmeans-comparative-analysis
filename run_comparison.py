"""
Главный скрипт для сравнения алгоритмов K-means
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

# Создаем папки для результатов
os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

# Импортируем наши алгоритмы
print("Импорт алгоритмов...")
try:
    from src.classic_kmeans import KMeansClassic
    from src.kmeans_plusplus import KMeansPlusPlus
    from src.minibatch_kmeans import MiniBatchKMeans
    print("✅ Алгоритмы успешно импортированы")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Проверьте файлы в папке src/")
    exit()

def generate_simple_data():
    """Генерация простых тестовых данных"""
    np.random.seed(42)
    
    # Три четких кластера
    cluster1 = np.random.randn(150, 2) * 0.5 + np.array([0, 0])
    cluster2 = np.random.randn(150, 2) * 0.5 + np.array([4, 4])
    cluster3 = np.random.randn(150, 2) * 0.5 + np.array([-4, 4])
    
    # Два перекрывающихся кластера
    cluster4 = np.random.randn(150, 2) * 1.0 + np.array([0, -4])
    cluster5 = np.random.randn(150, 2) * 1.0 + np.array([2, -4])
    
    return {
        'clear_clusters': np.vstack([cluster1, cluster2, cluster3]),
        'overlap_clusters': np.vstack([cluster4, cluster5])
    }

def test_algorithm(name, model, X, n_clusters):
    """Тестирование одного алгоритма"""
    print(f"\n  {name}:")
    
    # Измеряем время
    start_time = time.time()
    model.fit(X)
    end_time = time.time()
    
    # Вычисляем инерцию
    inertia = model.inertia_
    
    # Считаем размеры кластеров
    cluster_sizes = [np.sum(model.labels_ == i) for i in range(n_clusters)]
    
    return {
        'name': name,
        'time': end_time - start_time,
        'inertia': inertia,
        'iterations': model.n_iter_,
        'model': model,
        'cluster_sizes': cluster_sizes
    }

def plot_results(X, results, title, filename):
    """Визуализация результатов"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Исходные данные
    axes[0, 0].scatter(X[:, 0], X[:, 1], alpha=0.6, s=30, color='gray')
    axes[0, 0].set_title("Исходные данные")
    axes[0, 0].set_xlabel("Feature 1")
    axes[0, 0].set_ylabel("Feature 2")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Результаты каждого алгоритма
    colors = ['red', 'green', 'blue', 'orange']
    
    for idx, result in enumerate(results):
        row = (idx + 1) // 2
        col = (idx + 1) % 2
        
        model = result['model']
        labels = model.labels_
        n_clusters = len(model.centroids)
        
        # Рисуем каждый кластер своим цветом
        for cluster_id in range(n_clusters):
            cluster_points = X[labels == cluster_id]
            if len(cluster_points) > 0:
                axes[row, col].scatter(
                    cluster_points[:, 0], cluster_points[:, 1],
                    color=colors[cluster_id], alpha=0.6, s=30,
                    label=f'Кластер {cluster_id}'
                )
        
        # Рисуем центроиды
        axes[row, col].scatter(
            model.centroids[:, 0], model.centroids[:, 1],
            color='black', marker='X', s=200, label='Центроиды'
        )
        
        axes[row, col].set_title(
            f"{result['name']}\n"
            f"Время: {result['time']:.3f}с\n"
            f"Инерция: {result['inertia']:.2f}"
        )
        axes[row, col].set_xlabel("Feature 1")
        axes[row, col].set_ylabel("Feature 2")
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'results/plots/{filename}', dpi=150, bbox_inches='tight')
    plt.close(fig)  # Закрываем график чтобы не показывался

def save_summary(results, dataset_name, filename):
    """Сохранение сводной таблицы"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА: {dataset_name}\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"{'Алгоритм':<20} {'Время (с)':<12} {'Инерция':<15} {'Итерации':<10} {'Размеры кластеров':<30}\n")
        f.write("-"*87 + "\n")
        
        for result in results:
            sizes_str = str(result['cluster_sizes'])
            f.write(f"{result['name']:<20} {result['time']:<12.4f} "
                   f"{result['inertia']:<15.2f} {result['iterations']:<10} {sizes_str:<30}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("ВЫВОДЫ:\n")
        
        # Находим лучший алгоритм по инерции
        best_by_inertia = min(results, key=lambda x: x['inertia'])
        fastest = min(results, key=lambda x: x['time'])
        
        f.write(f"1. Лучшее качество (минимальная инерция): {best_by_inertia['name']} ({best_by_inertia['inertia']:.2f})\n")
        f.write(f"2. Самый быстрый: {fastest['name']} ({fastest['time']:.3f}с)\n")
        f.write("3. Classic K-means: простейший, но менее стабильный\n")
        f.write("4. K-means++: лучшее качество за счет улучшенной инициализации\n")
        f.write("5. Mini-Batch: быстрее на больших данных, немного хуже качество\n")

def main():
    """Основная функция"""
    print("="*70)
    print("КОМПАРАТИВНЫЙ АНАЛИЗ АЛГОРИТМОВ K-MEANS")
    print("="*70)
    
    # Генерируем данные
    print("\n📊 Генерация тестовых данных...")
    data = generate_simple_data()
    
    all_results = {}
    
    # Тестируем на каждом типе данных
    for data_name, X in data.items():
        print(f"\n{'='*50}")
        print(f"ТЕСТИРОВАНИЕ НА ДАННЫХ: {data_name}")
        print(f"Размер данных: {X.shape}")
        print(f"{'='*50}")
        
        # Определяем количество кластеров
        if 'clear' in data_name:
            n_clusters = 3
        else:
            n_clusters = 2
        
        # Создаем модели
        algorithms = [
            ("Classic K-means", KMeansClassic(n_clusters=n_clusters, random_state=42)),
            ("K-means++", KMeansPlusPlus(n_clusters=n_clusters, random_state=42)),
            ("Mini-Batch K-means", MiniBatchKMeans(n_clusters=n_clusters, batch_size=100, random_state=42))
        ]
        
        results = []
        for name, model in algorithms:
            result = test_algorithm(name, model, X, n_clusters)
            results.append(result)
            print(f"    Время: {result['time']:.3f}с, Инерция: {result['inertia']:.2f}, "
                  f"Итераций: {result['iterations']}")
        
        # Визуализируем
        plot_title = f"Сравнение алгоритмов на данных: {data_name}"
        plot_filename = f"comparison_{data_name}.png"
        plot_results(X, results, plot_title, plot_filename)
        
        # Сохраняем результаты
        summary_filename = f"results/tables/summary_{data_name}.txt"
        save_summary(results, data_name, summary_filename)
        
        all_results[data_name] = results
    
    # Создаем финальный отчет
    create_final_report(all_results)
    
    print("\n" + "="*70)
    print("✅ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print("="*70)
    print("\n📁 Результаты сохранены:")
    print("   Графики: results/plots/")
    print("   Таблицы: results/tables/")
    print("\n📊 Для просмотра графиков откройте папку results/plots/")

def create_final_report(all_results):
    """Создание финального отчета"""
    report_file = "results/tables/final_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ФИНАЛЬНЫЙ ОТЧЕТ: КОМПАРАТИВНЫЙ АНАЛИЗ K-MEANS\n")
        f.write("="*70 + "\n\n")
        
        for data_name, results in all_results.items():
            f.write(f"\nДАННЫЕ: {data_name}\n")
            f.write("-"*50 + "\n")
            
            # Сортируем по инерции (лучшее качество первое)
            sorted_results = sorted(results, key=lambda x: x['inertia'])
            
            for i, result in enumerate(sorted_results):
                rank = i + 1
                f.write(f"{rank}. {result['name']}: "
                       f"Инерция={result['inertia']:.2f}, "
                       f"Время={result['time']:.3f}с\n")
            
            f.write("\n")
        
        # Общие выводы
        f.write("\n" + "="*70 + "\n")
        f.write("ОБЩИЕ ВЫВОДЫ И РЕКОМЕНДАЦИИ:\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. Classic K-means:\n")
        f.write("   - Плюсы: Простота реализации, быстрая работа на малых данных\n")
        f.write("   - Минусы: Нестабильность, зависит от случайной инициализации\n")
        f.write("   - Рекомендация: Использовать для прототипирования\n\n")
        
        f.write("2. K-means++:\n")
        f.write("   - Плюсы: Лучшее качество кластеризации, стабильность\n")
        f.write("   - Минусы: Медленнее инициализация\n")
        f.write("   - Рекомендация: Использовать для финального анализа данных\n\n")
        
        f.write("3. Mini-Batch K-means:\n")
        f.write("   - Плюсы: Очень быстрая работа, экономия памяти\n")
        f.write("   - Минусы: Чуть хуже качество, требует настройки размера батча\n")
        f.write("   - Рекомендация: Использовать для больших объемов данных\n\n")
        
        f.write("ВЫВОД: Выбор алгоритма зависит от задачи:\n")
        f.write("- Для быстрого прототипа: Classic K-means\n")
        f.write("- Для точной кластеризации: K-means++\n")
        f.write("- Для больших данных: Mini-Batch K-means\n")

if __name__ == "__main__":
    main()