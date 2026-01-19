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

def plot_clusters_2d(X, labels, centroids, title="Кластеризация K-means", 
                    xlabel="Признак 1", ylabel="Признак 2", 
                    save_path=None, show_plot=True):
    """
    Визуализация результатов кластеризации для 2D данных
    
    Параметры:
    ----------
    X : array, данные (должны быть 2D)
    labels : array, метки кластеров
    centroids : array, центроиды
    title : str, заголовок графика
    xlabel, ylabel : str, подписи осей
    save_path : str, путь для сохранения (опционально)
    show_plot : bool, показывать ли график (True) или только сохранять (False)
    """
    plt.figure(figsize=(12, 10))
    
    # Цвета для кластеров
    colors = plt.cm.tab10(np.linspace(0, 1, len(centroids)))
    
    # Рисуем точки
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       color=colors[i], alpha=0.6, s=50, 
                       label=f'Кластер {i+1} ({len(cluster_points)} точек)',
                       edgecolors='white', linewidth=0.5)
    
    # Рисуем центроиды
    plt.scatter(centroids[:, 0], centroids[:, 1], 
                color='black', marker='X', s=300, label='Центроиды', 
                linewidths=2, zorder=10)
    
    # Подписываем центроиды
    for i, centroid in enumerate(centroids):
        plt.text(centroid[0], centroid[1], f' C{i+1}', fontsize=12, 
                fontweight='bold', verticalalignment='center')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        # Создаем папку если нет
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  График кластеров сохранен: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_convergence(inertia_history, title="Сходимость алгоритма", 
                    save_path=None, show_plot=True):
    """
    Визуализация сходимости алгоритма (изменение инерции)
    
    Параметры:
    ----------
    inertia_history : list, история изменения инерции
    title : str, заголовок графика
    save_path : str, путь для сохранения (опционально)
    show_plot : bool, показывать ли график
    """
    plt.figure(figsize=(10, 6))
    
    # Если история слишком длинная, прореживаем для читаемости
    if len(inertia_history) > 50:
        step = max(1, len(inertia_history) // 50)
        indices = list(range(0, len(inertia_history), step))
        if indices[-1] != len(inertia_history) - 1:
            indices.append(len(inertia_history) - 1)
        x = indices
        y = [inertia_history[i] for i in indices]
    else:
        x = range(len(inertia_history))
        y = inertia_history
    
    plt.plot(x, y, marker='o', markersize=6, linewidth=2, 
             color='steelblue', markeredgecolor='white', markeredgewidth=1)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Итерация', fontsize=12)
    plt.ylabel('Инерция (within-cluster sum of squares)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Добавляем аннотацию с финальным значением
    if inertia_history:
        final_inertia = inertia_history[-1]
        plt.annotate(f'Финальная инерция: {final_inertia:.2f}', 
                    xy=(0.98, 0.02), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show_plot:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()

def plot_convergence_comparison(results_dict, title="Сравнение сходимости алгоритмов", 
                               save_path=None, show_plot=True):
    """
    График сравнения сходимости алгоритмов (история изменения инерции)
    
    Параметры:
    ----------
    results_dict : dict
        Словарь с результатами: {'algorithm_name': {'model': model_obj, ...}, ...}
    title : str
        Заголовок графика
    save_path : str, optional
        Путь для сохранения графика
    show_plot : bool, optional
        Показывать ли график
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Цвета для алгоритмов
    
    for idx, (name, data) in enumerate(results_dict.items()):
        model = data.get('model', data)
        
        # Проверяем, есть ли история инерции у модели
        if hasattr(model, 'inertia_history') and model.inertia_history:
            history = model.inertia_history
            
            # Если история слишком длинная, берем каждую 2-ю точку для читаемости
            if len(history) > 50:
                step = max(1, len(history) // 50)
                indices = list(range(0, len(history), step))
                if indices[-1] != len(history) - 1:
                    indices.append(len(history) - 1)
                history = [history[i] for i in indices]
                x = indices
            else:
                x = range(len(history))
            
            plt.plot(x, history, 'o-', linewidth=2, markersize=5, 
                    color=colors[idx % len(colors)],
                    label=f"{name} (итер.: {model.n_iter_}, инерция: {model.inertia_:.2f})",
                    alpha=0.8)
        else:
            print(f" У алгоритма {name} нет истории инерции")
            # Если нет истории, рисуем точку с финальным значением
            if hasattr(model, 'inertia_'):
                plt.scatter([0], [model.inertia_], color=colors[idx % len(colors)], 
                          s=100, label=f"{name} (инерция: {model.inertia_:.2f})", 
                          zorder=10)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Итерация", fontsize=14)
    plt.ylabel("Инерция", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  График сходимости сохранен: {save_path}")
    
    if show_plot:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()

def evaluate_stability(model_class, X, n_runs=10, **kwargs):
    """
    Оценка стабильности алгоритма при многократных запусках
    
    Параметры:
    ----------
    model_class : class
        Класс модели (например, KMeansClassic)
    X : array
        Данные для обучения
    n_runs : int
        Количество запусков
    **kwargs : dict
        Дополнительные параметры для модели
    
    Возвращает:
    -----------
    mean_inertia : float, средняя инерция
    std_inertia : float, стандартное отклонение инерции
    mean_time : float, среднее время
    inertias : list, список всех инерций
    """
    inertias = []
    times = []
    
    print(f"  Оценка стабильности ({n_runs} запусков):")
    
    for i in range(n_runs):
        # Создаем модель с разным random_state для каждого запуска
        model = model_class(random_state=42 + i, **kwargs)
        
        start_time = time.time()
        model.fit(X)
        end_time = time.time()
        
        inertias.append(model.inertia_)
        times.append(end_time - start_time)
        
        print(f"  Запуск {i+1}/{n_runs}: инерция = {model.inertia_:.2f}, "
              f"время = {times[-1]:.3f}с")
    
    mean_inertia = np.mean(inertias)
    std_inertia = np.std(inertias)
    mean_time = np.mean(times)
    
    # Коэффициент вариации (в процентах)
    cv = (std_inertia / mean_inertia * 100) if mean_inertia > 0 else 0
    
    print(f"\n  Результаты {n_runs} запусков:")
    print(f"  Средняя инерция: {mean_inertia:.2f}")
    print(f"  Стандартное отклонение: {std_inertia:.2f}")
    print(f"  Коэффициент вариации: {cv:.1f}%")
    print(f"  Среднее время: {mean_time:.3f}с")
    
    return mean_inertia, std_inertia, mean_time, inertias

def test_hypothesis_1(results_dict):
    """
    Тестирование гипотезы 1: K-means++ лучше Classic K-means
    """
    classic = results_dict.get('Classic K-means')
    kmpp = results_dict.get('K-means++')
    
    if not classic or not kmpp:
        print("     Недостаточно данных для проверки гипотезы 1")
        return
    
    # Используем mean_inertia для стабильности или inertia для одиночных запусков
    classic_inertia = classic.get('mean_inertia', classic.get('inertia', 0))
    kmpp_inertia = kmpp.get('mean_inertia', kmpp.get('inertia', 0))
    
    improvement = ((classic_inertia - kmpp_inertia) / classic_inertia * 100 
                   if classic_inertia > 0 else 0)
    
    print(f"    Гипотеза 1: K-means++ лучше Classic K-means")
    print(f"      Classic K-means инерция: {classic_inertia:.2f}")
    print(f"      K-means++ инерция: {kmpp_inertia:.2f}")
    print(f"      Улучшение: {improvement:.1f}%")
    
    if kmpp_inertia < classic_inertia and improvement > 5:
        print(f"       ГИПОТЕЗА ПОДТВЕРЖДЕНА (улучшение >5%)")
    elif kmpp_inertia < classic_inertia:
        print(f"        ГИПОТЕЗА ЧАСТИЧНО ПОДТВЕРЖДЕНА (незначительное улучшение)")
    else:
        print(f"       ГИПОТЕЗА ОПРОВЕРГНУТА")

def test_hypothesis_2(results_dict, dataset_info=None):
    """
    Тестирование гипотезы 2: Mini-Batch быстрее но менее точен
    """
    classic = results_dict.get('Classic K-means')
    mb_name = 'Mini-Batch K-means' if 'Mini-Batch K-means' in results_dict else 'Mini-Batch K-means (оптимизированный)'
    mb = results_dict.get(mb_name)
    
    if not classic or not mb:
        print("     Недостаточно данных для проверки гипотезы 2")
        return None, None
    
    # Используем mean_time и mean_inertia для стабильности или time/inertia для одиночных
    classic_time = classic.get('mean_time', classic.get('time', 0))
    classic_inertia = classic.get('mean_inertia', classic.get('inertia', 0))
    mb_time = mb.get('mean_time', mb.get('time', 0))
    mb_inertia = mb.get('mean_inertia', mb.get('inertia', 0))
    
    # Правильно рассчитываем ускорение/замедление
    if mb_time > 0 and classic_time > 0:
        if mb_time < classic_time:
            speedup = classic_time / mb_time
            speedup_text = f"Ускорение: {speedup:.1f}x"
        else:
            slowdown = mb_time / classic_time
            speedup_text = f"Замедление: {slowdown:.1f}x"
    else:
        speedup_text = "Ускорение: N/A"
    
    accuracy_loss = ((mb_inertia - classic_inertia) / classic_inertia * 100 
                     if classic_inertia > 0 else 0)
    
    print(f"    Гипотеза 2: Mini-Batch быстрее но менее точен")
    print(f"      Classic K-means: {classic_time:.3f}с, инерция: {classic_inertia:.2f}")
    print(f"      {mb_name}: {mb_time:.3f}с, инерция: {mb_inertia:.2f}")
    print(f"      {speedup_text}, Потеря точности: {accuracy_loss:.1f}%")
    
    # Определяем тип данных по размеру
    if dataset_info and 'X' in dataset_info:
        n_samples = dataset_info['X'].shape[0]
    else:
        n_samples = 0
    
    if n_samples > 10000 or (dataset_info and 'большие' in str(dataset_info.get('name', '')).lower()):
        print(f"\n     Анализ для БОЛЬШИХ данных ({n_samples} точек):")
        if mb_time < classic_time:
            print(f"       ✓ Mini-Batch быстрее на больших данных!")
            if mb_inertia > classic_inertia:
                print(f"         Точность ниже на {accuracy_loss:.1f}% (ожидаемо для Mini-Batch)")
                print(f"       ✓ ГИПОТЕЗА 2 ПОДТВЕРЖДЕНА ДЛЯ БОЛЬШИХ ДАННЫХ!")
            else:
                print(f"       ✓ И качество лучше! Идеальный случай")
        else:
            print(f"       ✗ Mini-Batch не быстрее на больших данных")
    else:
        print(f"\n     Анализ для данных ({n_samples} точек):")
        if mb_time < classic_time and mb_inertia > classic_inertia:
            print(f"       ✓ ГИПОТЕЗА 2 ПОДТВЕРЖДЕНА")
        elif mb_time < classic_time:
            print(f"        Mini-Batch быстрее, но качество не хуже (неожиданно!)")
        else:
            print(f"        Mini-Batch медленнее (ожидаемо для малых данных)")
    
    return (classic_time / mb_time if mb_time > 0 else 0), accuracy_loss

def generate_recommendations(results_dict):
    """
    Генерация практических рекомендаций по выбору алгоритма
    """
    print("     Рекомендации по выбору алгоритма:")
    
    # Собираем все метрики
    algorithms = {}
    for name, data in results_dict.items():
        algorithms[name] = {
            'time': data.get('mean_time', data.get('time', 0)),
            'inertia': data.get('mean_inertia', data.get('inertia', 0))
        }
    
    # Рекомендация 1: Лучший по качеству (inertia)
    if algorithms:
        best_quality = min(algorithms.items(), key=lambda x: x[1]['inertia'])
        print(f"      1. Для максимального качества: {best_quality[0]}")
        print(f"         Инерция: {best_quality[1]['inertia']:.2f}")
    
    # Рекомендация 2: Самый быстрый
    if algorithms:
        fastest = min(algorithms.items(), key=lambda x: x[1]['time'])
        print(f"      2. Для скорости: {fastest[0]}")
        print(f"         Время: {fastest[1]['time']:.3f}с")
    
    # Рекомендация 3: Баланс скорости и качества
    if len(algorithms) >= 2:
        balanced = None
        best_score = float('inf')
        
        for name, metrics in algorithms.items():
            if metrics['time'] > 0 and metrics['inertia'] > 0:
                # Оценка баланса (чем меньше, тем лучше)
                balance_score = metrics['inertia'] * metrics['time']
                if balance_score < best_score:
                    best_score = balance_score
                    balanced = name
        
        if balanced:
            print(f"      3. Для баланса скорости и качества: {balanced}")
            print(f"         Время: {algorithms[balanced]['time']:.3f}с, "
                  f"Инерция: {algorithms[balanced]['inertia']:.2f}")
    
    print(f"      4. Общие рекомендации:")
    print(f"         • Classic K-means: для прототипирования и обучения")
    print(f"         • K-means++: для точной кластеризации и финального анализа")
    print(f"         • Mini-Batch K-means: для больших данных и интерактивного анализа")

def save_results_to_file(results, dataset_info, filename):
    """
    Сохранение результатов в текстовый файл
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА\n")
        f.write(f"Датасет: {dataset_info['name']}\n")
        from datetime import datetime
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Характеристики датасета:\n")
        f.write(f"  Размер: {dataset_info['X'].shape}\n")
        f.write(f"  Кластеров: {dataset_info['n_clusters']}\n\n")
        
        f.write("Результаты:\n")
        f.write("-"*80 + "\n")
        
        if 'inertia' in results[0]:  # одиночные запуски
            f.write(f"{'Алгоритм':<30} {'Время (с)':<12} {'Инерция':<15} {'Итерации':<10}\n")
            f.write("-"*80 + "\n")
            
            for result in results:
                f.write(f"{result['name']:<30} {result['time']:<12.4f} "
                       f"{result['inertia']:<15.2f} {result.get('iterations', 0):<10}\n")
        else:  # стабильность
            f.write(f"{'Алгоритм':<30} {'Ср. время (с)':<14} {'Ср. инерция':<15} "
                   f"{'Стд. отклонение':<15} {'Запусков':<10}\n")
            f.write("-"*80 + "\n")
            
            for result in results:
                f.write(f"{result['name']:<30} {result['mean_time']:<14.4f} "
                       f"{result['mean_inertia']:<15.2f} {result['std_inertia']:<15.2f} "
                       f"{result['stability_runs']:<10}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"   Результаты сохранены: {filename}")

def plot_algorithm_comparison(results, dataset_name, save_path=None, show_plot=False):
    """
    Bar-plot для сравнения инерции и времени алгоритмов
    """
    algorithms = [r['name'] for r in results]
    
    # Извлекаем данные
    if 'inertia' in results[0]:  # одиночные запуски
        inertias = [r['inertia'] for r in results]
        times = [r['time'] for r in results]
        title_suffix = " (одиночный запуск)"
    else:  # стабильность
        inertias = [r['mean_inertia'] for r in results]
        inertias_std = [r['std_inertia'] for r in results]
        times = [r['mean_time'] for r in results]
        title_suffix = " (средние за 10 запусков)"
    
    # Создаем график
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # График инерции
    x = np.arange(len(algorithms))
    width = 0.35
    
    if 'inertias_std' in locals() and inertias_std:
        bars1 = ax1.bar(x, inertias, width, yerr=inertias_std, 
                       capsize=5, color='skyblue', label='Инерция')
    else:
        bars1 = ax1.bar(x, inertias, width, color='skyblue', label='Инерция')
    
    ax1.set_xlabel('Алгоритм', fontsize=12)
    ax1.set_ylabel('Инерция (чем меньше, тем лучше)', fontsize=12)
    ax1.set_title(f'Качество кластеризации\n{dataset_name}{title_suffix}', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=10)
    ax1.legend(fontsize=11)
    
    # Добавляем значения на столбцы
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # График времени
    bars2 = ax2.bar(x, times, width, color='lightcoral', label='Время (с)')
    ax2.set_xlabel('Алгоритм', fontsize=12)
    ax2.set_ylabel('Время выполнения (секунды)', fontsize=12)
    ax2.set_title(f'Время выполнения\n{dataset_name}{title_suffix}', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=10)
    ax2.legend(fontsize=11)
    
    # Добавляем значения на столбцы
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Bar-plot сохранен: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig