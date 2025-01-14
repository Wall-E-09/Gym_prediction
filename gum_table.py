import pandas as pd
import random
from datetime import datetime, timedelta

num_visitors = 1000

def time_to_minutes(time_str):
    hour, minute = map(int, time_str.split('.'))
    return hour * 60 + minute

def generate_time_and_day():
    day_of_week = random.choice(['Понеділок', 'Вівторок', 'Середа', 'Четвер', 'Пʼятниця', 'Субота', 'Неділя'])
    
    if day_of_week in ['Субота', 'Неділя']:  # Вихідні дні
        opening_time = 9
        closing_time = 20
    else:  # Будні дні
        opening_time = 8 
        closing_time = 22 
    
    arrival_hour = random.randint(opening_time, closing_time)
    arrival_minute = random.randint(0, 59)
    
    if arrival_hour == closing_time and arrival_minute > 0:
        arrival_hour = closing_time - 1
        arrival_minute = random.randint(0, 59)

    arrival_time = f"{arrival_hour:02d}.{arrival_minute:02d}"
    
    max_duration = (closing_time * 60) - time_to_minutes(arrival_time)  
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
    
    if arrival_time is None:
        continue
    
    arrival_times.append(arrival_time)
    departure_times.append(departure_time)
    days_of_week.append(day_of_week)
    
    age = random.randint(6, 80)
    categories.append('Дитина' if age < 18 else 'Пенсіонер' if age >= 65 else random.choice(['Чоловік', 'Дівчина']))
    
    arrival_minutes = time_to_minutes(arrival_time)
    departure_minutes = time_to_minutes(departure_time)
    duration = departure_minutes - arrival_minutes  
    durations.append(duration)
    
    cardio.append('Так' if random.random() < 0.3 else 'Ні')

data = pd.DataFrame({
    'Вік': [random.randint(6, 80) for _ in range(num_visitors)],
    'Час прийшли': arrival_times,
    'Час вийшли': departure_times,
    'Час перебування (хв)': durations,
    'Категорія': categories,
    'День тижня': days_of_week,
    'Кардіо': cardio
})

file_path = r'C:\Users\Home\Documents\GitHub\KPRS_RGR\gum_visitors.csv'
data.to_csv(file_path, index=False)

file_path  
