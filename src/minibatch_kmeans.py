import numpy as np
from .base_kmeans import BaseKMeans

class MiniBatchKMeans(BaseKMeans):
    """Mini-Batch K-means для работы с большими данными"""
    
    def __init__(self, n_clusters=3, max_iter=100, batch_size=100, 
                 tol=1e-4, random_state=None, n_init=10):
        super().__init__(n_clusters, max_iter, tol, random_state)
        self.batch_size = batch_size
        self.n_init = n_init  # Количество запусков с разной инициализацией
    
    def _initialize_centroids(self, X):
        # Используем случайную инициализацию
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]
    
    def fit(self, X):
        """
        Обучение Mini-Batch K-means
        
        Особенности:
        - Использует мини-батчи (подвыборки) данных
        - Инкрементальное обновление центроидов
        - Быстрее на больших данных
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Лучший результат из нескольких запусков
        best_inertia = float('inf')
        best_centroids = None
        best_labels = None
        
        for init in range(self.n_init):
            # Инициализация центроидов
            self.centroids = self._initialize_centroids(X)
            
            # Для инкрементального обновления центроидов
            centroid_counts = np.zeros(self.n_clusters)
            
            for iteration in range(self.max_iter):
                # Выбираем случайный мини-батч
                indices = np.random.choice(n_samples, 
                                          min(self.batch_size, n_samples), 
                                          replace=False)
                X_batch = X[indices]
                
                # Находим ближайшие центроиды для батча
                distances = self._compute_distances(X_batch, self.centroids)
                batch_labels = self._assign_clusters(distances)
                
                # Инкрементальное обновление центроидов
                for i in range(len(X_batch)):
                    label = batch_labels[i]
                    centroid_counts[label] += 1
                    # Скорость обучения уменьшается с количеством наблюдений
                    learning_rate = 1.0 / centroid_counts[label]
                    self.centroids[label] = (1 - learning_rate) * self.centroids[label] + \
                                           learning_rate * X_batch[i]
                
                # Проверка сходимости (раз в 10 итераций)
                if iteration % 10 == 0:
                    all_labels = self.predict(X)
                    current_inertia = self._compute_inertia(X, all_labels, self.centroids)
                    
                    if iteration > 0:
                        inertia_change = abs(previous_inertia - current_inertia)
                        if inertia_change < self.tol:
                            break
                    
                    previous_inertia = current_inertia
            
            # Вычисляем финальную инерцию
            self.labels_ = self.predict(X)
            self.inertia_ = self._compute_inertia(X, self.labels_, self.centroids)
            
            # Сохраняем лучший результат
            if self.inertia_ < best_inertia:
                best_inertia = self.inertia_
                best_centroids = self.centroids.copy()
                best_labels = self.labels_.copy()
        
        # Сохраняем лучшие параметры
        self.centroids = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        
        return self