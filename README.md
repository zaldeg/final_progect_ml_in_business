# final_progect_ml_in_business

Итоговый проэкт курса "Машинное обучение в бизнесе"

Проэкт построен на базе данных: https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction
Используются пакеты:
- numpy
- sklearn
- pandas
- dill
- flask
- urlib
- json
- requests

Задача проекта прогнозировать удовлетворенность пассажиров авиалиний. Для возможного превентивного воздействия на них, в случае если они могут быть неудовлетворены.

Используемые признаки: 
Категориальные
- Type of Travel ('Personal Travel', 'Business travel')
- Gender ('Female', 'Male')
- Class ('Eco Plus', 'Business', 'Eco')
- Customer Type ('Loyal Customer', 'disloyal Customer')
- Inflight wifi service (0, 1, 2, 3, 4, 5)
- Departure/Arrival time convenient (0, 1, 2, 3, 4, 5)
- Ease of Online booking (0, 1, 2, 3, 4, 5)
- Gate location (0, 1, 2, 3, 4, 5)
- Food and drink (0, 1, 2, 3, 4, 5)
- Online boarding (0, 1, 2, 3, 4, 5)
- Seat comfort (0, 1, 2, 3, 4, 5)
- Inflight entertainment (0, 1, 2, 3, 4, 5)
- On-board service (0, 1, 2, 3, 4, 5)
- Leg room service (0, 1, 2, 3, 4, 5)
- Baggage handling (0, 1, 2, 3, 4, 5)
- Checkin service (0, 1, 2, 3, 4, 5)
- Inflight service (0, 1, 2, 3, 4, 5)
- Cleanliness (0, 1, 2, 3, 4, 5)
- Ease of Online booking (0, 1, 2, 3, 4, 5)

Вещественные:
- Age
- Flight Distance
- Departure Delay in Minutes
- Arrival Delay in Minutes

В проекте использован sklearn.ensemble.GradientBoostingClassifier

Можно использовать любое сочетание признаков. Все неправильно указанные или отсутствующие признаки будут заменяться моделью на "медианное" значение для вещественных признаков и на "самое часто встречающееся" для категориальных значений. Если отослать пустой json то модель вернет предсказание для "самого среднего пассажира". 

Данные принимаются flask сервером в виде POST запроса содержащего json файл вида.
``` 
{'Gender': 'Female',
 'Customer Type': 'Loyal Customer',
 'Age': 26,
 'Type of Travel': 'Business travel',
 'Class': 'Business',
 'Flight Distance': 2123,
 'Inflight wifi service': 3,
 'Departure/Arrival time convenient': 3,
 'Gate location': 3,
 'Food and drink': 4,
 'Online boarding': 4,
 'Seat comfort': 4,
 'Inflight entertainment': 4,
 'On-board service': 5,
 'Leg room service': 3,
 'Baggage handling': 4,
 'Checkin service': 5,
 'Inflight service': 4,
 'Cleanliness': 4,
 'Departure Delay in Minutes': 49,
 'Arrival Delay in Minutes': 51.0}
 ```
Вернется json в котором в поле 'predictions' будет содержаться список с предсказанием.
Каждое значение в отправляемом словаре можно заменить на список(отослать не один пример а множество), в таком случае сервер вернет список предсказний по каждому примеру.

## Запуска проекта.
- Клонируем проект
```
https://github.com/zaldeg/final_progect_ml_in_business.git
```
- запускаем сревер run_server.py
- step3.ipynb дает возможность сдлеать запрос для проверки сервера.

## дополнительно
- step1.ipynb содержит построение Pipeline модели.
- step2.ipynb содержит проверку сохраненной Pipeline модели и ее качества.
