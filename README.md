# Object Tracking Home assignment

### Установка зависимостей
```
pip install -r requirements.txt
```

### Запуск сервера
Или настройте запуск файла fastapi_server.py как приведено на скриншоте ниже 
![fastapi.png](info/fastapi.png)

или командой в терминале
```
python3 -m uvicorn fastapi_server:app --reload --port 8001 
```

### Постановка задачи

Реализуйте методы tracker_soft и tracker_strong в скрипте fastapi_server.py,
придумайте, обоснуйте и реализуйте метод для оценки качества разработанных трекеров.
Сравните результаты tracker_soft и tracker_strong для 5, 10, 20 объектов и различных 
значений random_range и bb_skip_percent
(информацию о генерации данных читай в пункте "Тестирование"). Напишите отчёт. 
В отчете необходимо в свободном стиле привести описание методов tracker_soft, 
tracker_strong, метода оценки качества трекеров, привести сравнительную таблицу 
реализованных трекеров, сделать вывод.  
Бонусом можете выписать найденные баги в текущем проекте.

### Тестирование
Для тестирования можно воспользоваться скриптом create_track.py. Скрипт генерирует
информацию об объектах и их треках. Скопируйте вывод в новый скрипт track_n.py и
скорректируйте импорт в fastapi_server.py
```
from track_n import track_data, country_balls_amount
```
Что стоит менять в скрипте create_track.py:  
**tracks_amount**: количество объектов  
**random_range**: на сколько пикселей рамка объектов может ложно смещаться (эмуляция не идеальной детекции)  
**bb_skip_percent**: с какой вероятностью объект на фрейме может быть не найдет детектором


## Отчёт

### Soft Tracker

Был реализован следующий подход:

- (1) для каждого нового кадра сначала строится соответствие между сохранёнными треками и пришедшими детекциями (Венгерским алгоритмом решается задача о назначениях для матрицы значений IoU между последними bbox-ами треков и новыми детекциями);
- (2) несматченные с треками детекции (либо когда число детекций больше количества треков, либо нулевое значение IoU с оптимальным треком из предыдущего пункта) жадным образом ставятся в соответствие с ближайшим по расстоянию между центрами bbox-ов треком из оставшихся;
- (3) для оставшихся без трека детекций инициализируются новые треки.

### Strong Tracker

Отличие от Soft трекера в том, что на (1) шаге считается другая матрица расстояний: не IoU, а средневзвешенная комбинация расстояния между центрами bbox-ов и расстояния между эмбеддингами. Эмбеддинги получаем с помощью предобученной на ImageNet модели ResNet34.

### Метод оценки качества

Оценка качества трекинга осуществляется с помощью набора метрик:

1. **diff_match_val** отвечает на вопрос, насколько разнообразны предсказанные треки после построения соответствия между gt-треками и pred_ids. Если разным gt-трекам соответствует один и тот же предсказанный id, то метрика будет снижена. Чем больше, тем лучше
2. **recall** $-$ насколько сильно (в среднем по трекам) покрываем gt-треки соответствующими pred_id? Чем больше, тем лучше
3. **precision** $-$ насколько часто (в среднем по трекам) соответствующие pred_id находятся в gt-треке, с которым построили соответствие? Чем больше, тем лучше
4. **f1-score** $-$ среднее гармоническое precision и recall. Чем больше, тем лучше
5. **none_freq** $-$ средняя частотность предсказания None trk_id (такое может случиться только при отсутствии детекции).Чем больше, тем хуже

! Для метрик (1)-(4) считаем, что gt_trk_id соответствует наиболее частотный trk_id из предсказанных на bbox-ах из gt_trk_id.


### Замеры метрик для разных параметров

В таблице ниже представлены значения метрик на примерах, сгенерированных с различными параметрами (tracks_amount=obj_num, random_range, bb_skip_percent), для двух реализованных трекеров.

| **obj_num** 	      | **random_range** 	 | **bb_skip_percent** 	 |**tracker** 	 |**diff_match_val(%)** 	 |**P(%)**  	 | **R(%)** 	 | **F1(%)** 	 | **none_freq(%)** 	 |
|---------------|-----------------|--------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
|5 | 10 | 0.25| Soft | 60.00 | 47.27 | 40.81 | 43.80 |25.43 |
|5 | 10 | 0.25| Strong | 100.00 | 95.27 | 40.81 | 43.80 |25.43 |
|5 | 10 | 0.50| Soft | 80.00 | 50.94 | 53.03 | 51.96 | 46.87 |
|5 | 10 | 0.50| Strong | 60.00 | 36.91 | 48.31 | 41.85 | 46.87 |
|5 | 20 | 0.25| Soft | 80.00 | 54.80 | 45.55 | 49.75 | 22.66 |
|5 | 20 | 0.25| Strong | 100.00 | 100.00 | 77.34 | 87.22 | 22.66 |
|10 | 10 | 0.25| Soft | 60.00 | 32.40 | 33.63 | 33.01 | 23.49 |
|10 | 10 | 0.25| Strong | 90.00 | 60.75 | 48.03 | 53.64 | 23.49 |
|10 | 10 | 0.50| Soft | 30.00 | 23.96 | 52.98 | 33.00 | 50.08 |
|10 | 10 | 0.50| Strong | 40.00 | 31.82 | 54.82 | 40.27 | 50.08 |
|10 | 20 | 0.25| Soft | 90.00 | 45.95 | 39.75 | 42.62 | 26.72 |
|10 | 20 | 0.25| Strong | 100.00 | 73.32 | 54.64 | 62.61 | 26.72 |
|20 | 10 | 0.25| Soft | 50.00 | 22.83 | 30.18 | 25.99 | 24.50 |
|20 | 10 | 0.25| Strong | 60.00 | 39.92 | 39.13 | 39.52 | 24.50 |
|20 | 10 | 0.50| Soft | 10.00 | 7.55 | 52.05 | 13.19 | 51.70 |
|20 | 10 | 0.50| Strong | 5.00 | 5.00 | 51.70 | 9.12 | 51.70 |
|20 | 20 | 0.25| Soft | 50.00 | 24.55 | 34.85 | 28.81 | 26.47 |
|20 | 20 | 0.25| Strong | 70.00 | 44.68 | 44.39 | 44.53 | 26.47 |

### Наблюдения и выводы

- в большинстве случаев при одинаковых параметрах генерации для StrongTracker метрики лучше, чем для Soft => с использованием визуальной информации (эмбеддингов) трекинг работает лучше, чем если использовать только геометрию
- метрики для Strong хуже, чем для Soft, в замерах с большим значением параметра, задающего частоту потери детекций (скорее всего, это связано с большим количеством None в предсказываемых треках)
- с увеличением bb_skip_percent наблюдается увеличение значения none_freq
- странное наблюдение: при увеличении random_range метрики становятся лучше