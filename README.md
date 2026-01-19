\# Сравнительный анализ алгоритмов K-means



\## Описание проекта

Полный эксперимент по сравнительному анализу трех алгоритмов кластеризации:

1\. Classic K-means - классический алгоритм

2\. K-means++ - улучшенная инициализация  

3\. Mini-Batch K-means - оптимизирован для больших данных



\## Структура проекта

kmeans-comparative-analysis/

├── run\_full\_experiment.py    # Главный скрипт эксперимента

├── src/                      # Реализации алгоритмов

│   ├── classic\_kmeans.py

│   ├── kmeans\_plusplus.py

│   ├── minibatch\_kmeans.py

│   ├── base\_kmeans.py

│   └── utils.py

├── tests/                    # Тесты

│   ├── test\_all\_algorithms.py

│   └── test\_run.py

├── results/                  # Результаты эксперимента

│   ├── plots/               # Графики

│   └── tables/              # Текстовые отчеты

└── README.md                # Этот файл



\## Запуск эксперимента

python run\_full\_experiment.py



\## Результаты эксперимента

Результаты сохранены в папке results/:

\- Графики сходимости (results/plots/convergence\_\*.png) - 6 файлов

\- Bar-plot сравнения (results/plots/bar\_comparison\_\*.png) - 6 файлов

\- Таблицы результатов (results/tables/\*.txt) - 13 файлов

\- Финальный отчет (results/tables/final\_summary.txt) - основные выводы



\## Проверенные гипотезы

1\. Гипотеза 1: K-means++ дает лучшие результаты чем Classic K-means

2\. Гипотеза 2: Mini-Batch K-means быстрее но менее точен на больших данных



\## Требования

\- Python 3.8+

\- NumPy

\- Matplotlib

\- scikit-learn



\## Автор

Заволожина Дарья

