import numpy as np
from .base_kmeans import BaseKMeans

class KMeansPlusPlus(BaseKMeans):
    """K-means++ с улучшенной инициализацией центроидов"""
    
    def _initialize_centroids(self, X):
        """
        Инициализация центроидов по алгоритму K-means++
        
        Шаги:
        1. Выбрать первый центроид случайно
        2. Для каждой точки вычислить расстояние до ближайшего центроида
        3. Выбрать следующий центроид с вероятностью, пропорциональной квадрату расстояния
        4. Повторять шаги 2-3 пока не выберем k центроидов
        """
        n_samples, n_features = X.shape
        
        # 1. Первый центроид выбираем случайно
        centroids = [X[np.random.randint(n_samples)]]
        
        # 2. Выбираем оставшиеся k-1 центроидов
        for _ in range(1, self.n_clusters):
            # Вычисляем квадраты расстояний от каждой точки до ближайшего центроида
            distances_sq = np.zeros(n_samples)
            for i, x in enumerate(X):
                # Находим минимальное расстояние до уже выбранных центроидов
                min_dist = min([np.sum((x - c) ** 2) for c in centroids])
                distances_sq[i] = min_dist
            
            # Преобразуем расстояния в вероятности
            probabilities = distances_sq / distances_sq.sum()
            
            # Выбираем следующий центроид согласно вероятностям
            cumulative_prob = np.cumsum(probabilities)
            r = np.random.rand()
            
            for j, cp in enumerate(cumulative_prob):
                if r < cp:
                    centroids.append(X[j])
                    break
        
        return np.array(centroids)