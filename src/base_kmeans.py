import numpy as np
from abc import ABC, abstractmethod

class BaseKMeans(ABC):
    """Базовый класс для всех алгоритмов K-means"""
    
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=None):
        """
        Инициализация базового класса
        
        Параметры:
        ----------
        n_clusters : int, default=3
            Количество кластеров
        max_iter : int, default=100
            Максимальное количество итераций
        tol : float, default=1e-4
            Точность для проверки сходимости
        random_state : int, default=None
            Seed для воспроизводимости результатов
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.inertia_history = []  # История инерции
        
        if random_state is not None:
            np.random.seed(random_state)
    
    @abstractmethod
    def _initialize_centroids(self, X):
        """Инициализация центроидов (абстрактный метод)"""
        pass
    
    def _compute_distances(self, X, centroids):
        """Вычисление расстояний от точек до центроидов"""
        # distances[i, j] = расстояние от точки i до центроида j
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(centroids):
            # Евклидово расстояние
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        return distances
    
    def _assign_clusters(self, distances):
        """Назначение меток кластеров (ближайший центроид)"""
        return np.argmin(distances, axis=1)
    
    def _compute_centroids(self, X, labels):
        """Пересчет центроидов как среднего точек в кластере"""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            # Проверяем, что в кластере есть точки
            if np.sum(labels == i) > 0:
                centroids[i] = X[labels == i].mean(axis=0)
        return centroids
    
    def _compute_inertia(self, X, labels, centroids):
        """Вычисление инерции (within-cluster sum of squares)"""
        inertia = 0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                # Сумма квадратов расстояний до центроида
                inertia += np.sum((cluster_points - centroids[i]) ** 2)
        return inertia
    
    def fit(self, X):
        """
        Обучение модели на данных X
        
        Параметры:
        ----------
        X : array-like, shape (n_samples, n_features)
            Входные данные
            
        Возвращает:
        -----------
        self : object
            Обученная модель
        """
        X = np.array(X)
        
        # 1. Инициализация центроидов
        self.centroids = self._initialize_centroids(X)
        
        # Очищаем историю инерции перед началом обучения
        self.inertia_history = []  # <-- ОЧИЩАЕМ ИСТОРИЮ
        
        for iteration in range(self.max_iter):
            # 2. Вычисление расстояний
            distances = self._compute_distances(X, self.centroids)
            
            # 3. Назначение кластеров
            labels = self._assign_clusters(distances)
            
            # 4. Пересчет центроидов
            new_centroids = self._compute_centroids(X, labels)
            
            # 5. Проверка сходимости (изменение центроидов)
            centroid_shift = np.sqrt(np.sum((new_centroids - self.centroids) ** 2))
            
            # 6. Обновление параметров
            self.centroids = new_centroids
            self.labels_ = labels
            self.n_iter_ = iteration + 1
            
            # 7. Вычисление инерции
            self.inertia_ = self._compute_inertia(X, labels, self.centroids)
            self.inertia_history.append(self.inertia_)  # сохраняем в историю
            
            # 8. Вывод информации (можно убрать позже)
            if iteration % 10 == 0:
                print(f"Итерация {iteration}: инерция = {self.inertia_:.4f}")
            
            # 9. Проверка условия остановки
            if centroid_shift < self.tol:
                print(f"Сходимость достигнута на итерации {iteration}")
                break
        
        if iteration == self.max_iter - 1:
            print(f"Достигнуто максимальное количество итераций ({self.max_iter})")
        
        return self
    
    def predict(self, X):
        """
        Предсказание кластеров для новых данных
        
        Параметры:
        ----------
        X : array-like, shape (n_samples, n_features)
            Новые данные
            
        Возвращает:
        -----------
        labels : array, shape (n_samples,)
            Метки кластеров
        """
        X = np.array(X)
        distances = self._compute_distances(X, self.centroids)
        return self._assign_clusters(distances)
    
    def fit_predict(self, X):
        """Обучение и предсказание на одних данных"""
        return self.fit(X).labels_