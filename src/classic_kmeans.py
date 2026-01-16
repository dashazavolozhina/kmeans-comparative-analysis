import numpy as np
from .base_kmeans import BaseKMeans

class KMeansClassic(BaseKMeans):
    """Классический K-means со случайной инициализацией"""
    
    def _initialize_centroids(self, X):
        """
        Случайная инициализация центроидов
        
        Параметры:
        ----------
        X : array-like, shape (n_samples, n_features)
            Входные данные
            
        Возвращает:
        -----------
        centroids : array, shape (n_clusters, n_features)
            Инициализированные центроиды
        """
        n_samples = X.shape[0]
        
        # Случайно выбираем k индексов из набора данных
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        
        # Возвращаем точки с выбранными индексами как начальные центроиды
        return X[indices]