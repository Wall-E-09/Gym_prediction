#file_path = r'C:\Users\Home\Documents\GitHub\KPRS_RGR\gum_visitors.csv'


import pandas as pd
import random
from datetime import datetime, timedelta

# Кількість відвідувачів
num_visitors = 2000

# Функція для перетворення часу у форматі "години.хвилини" в хвилини
def time_to_minutes(time_str):
    hour, minute = map(int, time_str.split('.'))
    return hour * 60 + minute

# Функція для визначення дня тижня та часу в залі
def generate_time_and_day():
    day_of_week = random.choice(['Понеділок', 'Вівторок', 'Середа', 'Четвер', 'Пʼятниця', 'Субота', 'Неділя'])
    
    if day_of_week in ['Субота', 'Неділя']:  # Вихідні дні
        opening_time = 9  # З 9:00
        closing_time = 20  # до 20:00
    else:  # Будні дні
        opening_time = 8  # З 8:00
        closing_time = 22  # до 22:00
    
    # Час прибуття відвідувача
    arrival_hour = random.randint(opening_time, closing_time)
    arrival_minute = random.randint(0, 59)
    
    # Перевірка, щоб не прийшли за годину до закриття
    if arrival_hour == closing_time and arrival_minute > 0:
        arrival_hour = closing_time - 1
        arrival_minute = random.randint(0, 59)

    arrival_time = f"{arrival_hour:02d}.{arrival_minute:02d}"
    
    # Час відвідувача в залі
    # Відвідувач може залишатися необмежену кількість часу, але має покинути зал до закриття
    # Тому час виходу не може перевищувати час закриття
    # Наприклад, відвідувач може тренуватися від 0 до 12 годин.
    
    max_duration = (closing_time * 60) - time_to_minutes(arrival_time)  # Максимум, до закриття
    duration = random.randint(0, max_duration)
    
    departure_minutes = time_to_minutes(arrival_time) + duration
    departure_hour = departure_minutes // 60
    departure_minute = departure_minutes % 60
    
    departure_time = f"{departure_hour:02d}.{departure_minute:02d}"
    
    return arrival_time, departure_time, day_of_week

arrival_times = []
departure_times = []
days_of_week = []
durations = []
categories = []
cardio = []

for _ in range(num_visitors):
    arrival_time, departure_time, day_of_week = generate_time_and_day()
    
    # Якщо відвідувач прийшов занадто пізно, пропускаємо його
    if arrival_time is None:
        continue
    
    arrival_times.append(arrival_time)
    departure_times.append(departure_time)
    days_of_week.append(day_of_week)
    
    # Генерація категорій за віком
    age = random.randint(6, 80)
    categories.append('Дитина' if age < 18 else 'Пенсіонер' if age >= 65 else random.choice(['Чоловік', 'Дівчина']))
    
    # Обрахунок часу перебування в залі
    arrival_minutes = time_to_minutes(arrival_time)
    departure_minutes = time_to_minutes(departure_time)
    duration = departure_minutes - arrival_minutes  # Різниця між часом прибуття та від'їзду
    durations.append(duration)
    
    # Додамо кардіо
    # Припустимо, що 30% відвідувачів займаються кардіо
    cardio.append('Так' if random.random() < 0.3 else 'Ні')

# Створення DataFrame
data = pd.DataFrame({
    'Вік': [random.randint(6, 80) for _ in range(num_visitors)],
    'Час прийшли': arrival_times,
    'Час вийшли': departure_times,
    'Час перебування (хв)': durations,
    'Категорія': categories,
    'День тижня': days_of_week,
    'Кардіо': cardio
})

# Збереження у файл gum_visitors.csv
file_path = r'C:\Users\Home\Documents\GitHub\KPRS_RGR\gum_visitors.csv'
data.to_csv(file_path, index=False)

file_path  # Повертаємо шлях до файлу
