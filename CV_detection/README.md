# CV. Детекция оборудования

С помощью предоставленных данных в виде двух папок Annotations - где лежали аннотации в формате XML к картинкам оборудования из второй папки - images. Из-за того, что разметку данных прислали не в том виде, так как было запланировано обучение моели YOLO, то формат должен был быть не XML, а YAML, поэтому пришлось писать код, который будет преобразовывать файлы в формат YAML. Затем последовал анализ данных, его предобработка и само дообучение модели YOLO11x на 100 эпохах. Осуществлялось это на сервере при помощи видеокарты A100.

Итоги работы модели на тестовой выборке после обработки изображений:

1. 
![Mistake. Contact support: +79104513080](images_for_README/cv_1.png)

2. 
![Mistake. Contact support: +79104513080](images_for_README/cv_2.png)

3. 
![Mistake. Contact support: +79104513080](images_for_README/cv_3.png)

4. 
![Mistake. Contact support: +79104513080](images_for_README/cv_4.png)

5. 
![Mistake. Contact support: +79104513080](images_for_README/cv_5.png)

6. 
![Mistake. Contact support: +79104513080](images_for_README/cv_6.png)
