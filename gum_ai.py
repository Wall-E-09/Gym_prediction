import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file_path_visitors = r'C:\Users\Home\Documents\GitHub\KPRS_RGR\gum_visitors.csv'
file_path_visitors_time = r'C:\Users\Home\Documents\GitHub\KPRS_RGR\gum_visitors_time.csv'
file_path_visitor_ratings = r'C:\Users\Home\Documents\GitHub\KPRS_RGR\visitor_data.csv'

data_visitors = pd.read_csv(file_path_visitors)
data_visitors_time = pd.read_csv(file_path_visitors_time)
data_visitor_ratings = pd.read_csv(file_path_visitor_ratings)

print("Структура даних для gum_visitors:")
print(data_visitors.head())  

print("Структура даних для gum_visitors_time:")
print(data_visitors_time.head())  

print("Структура даних для visitor_data:")
print(data_visitor_ratings.head())  

TOTAL_TRAINERS = 200
CARDIO_TRAINERS = 50
STRENGTH_TRAINERS = 150
CARDIO_DURATION = 20  

def time_to_minutes(time_str):
    if isinstance(time_str, str):
        hour, minute = map(int, time_str.split('.'))
        return hour * 60 + minute
    else:
        return 0  

def get_visitor_count_at_time(hour):
    visitor_count = data_visitors_time[data_visitors_time['Година'] == hour]['Кількість відвідувачів'].values
    return visitor_count[0] if len(visitor_count) > 0 else 0

def get_occupied_trainers(arrival_time, is_cardio, existing_visitors):
    arrival_minutes = time_to_minutes(arrival_time)
    occupied_cardio = sum(1 for _, visitor in existing_visitors.iterrows() 
                          if visitor['Кардіо'] == 'Так' and 
                             time_to_minutes(visitor['Час прийшли']) < arrival_minutes < time_to_minutes(visitor['Час вийшли']))
    
    occupied_strength = sum(1 for _, visitor in existing_visitors.iterrows() 
                            if visitor['Кардіо'] == 'Ні' and 
                               time_to_minutes(visitor['Час прийшли']) < arrival_minutes < time_to_minutes(visitor['Час вийшли']))
    
    return occupied_cardio, occupied_strength

X = []
y = []

data_visitors['Час прийшли хвилини'] = data_visitors['Час прийшли'].apply(time_to_minutes)
data_visitors['Час вийшли хвилини'] = data_visitors['Час вийшли'].apply(time_to_minutes)

for _, row in data_visitors.iterrows():
    arrival_time = row['Час прийшли']
    is_cardio = 1 if row['Кардіо'] == 'Так' else 0
    departure_time = row['Час вийшли']
    existing_visitors = data_visitors[(data_visitors['Час прийшли хвилини'] <= arrival_time) & 
                                      (data_visitors['Час вийшли хвилини'] > arrival_time)]
    
    occupied_cardio, occupied_strength = get_occupied_trainers(arrival_time, is_cardio, existing_visitors)
    
    comfortable = False
    if is_cardio:
        if occupied_cardio < CARDIO_TRAINERS:
            comfortable = True
    else:
        if occupied_strength < STRENGTH_TRAINERS:
            comfortable = True
    
    X.append([arrival_time, is_cardio])
    y.append(comfortable)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = np.array(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  

model.compile(loss=MeanSquaredError(), optimizer='adam', metrics=['accuracy'])  

model.fit(X_train, np.array(y_train), epochs=10, batch_size=16, validation_data=(X_test, np.array(y_test)))

def predict_occupied_trainers(hour, total_visitors, cardio_ratio=0.25):
    cardio_visitors = int(total_visitors * cardio_ratio)  
    strength_visitors = total_visitors - cardio_visitors  

    occupied_cardio = min(cardio_visitors, CARDIO_TRAINERS)
    occupied_strength = min(strength_visitors, STRENGTH_TRAINERS)
    
    extra_visitors = total_visitors - (CARDIO_TRAINERS + STRENGTH_TRAINERS)
    if extra_visitors > 0:
        extra_cardio = int(extra_visitors * cardio_ratio)
        extra_strength = extra_visitors - extra_cardio
        occupied_cardio += extra_cardio
        occupied_strength += extra_strength

    return occupied_cardio, occupied_strength

def get_comfort_level_cardio(occupied_cardio):
    if occupied_cardio > 50:
        return "Некомфортно"
    return "Комфортно"

def get_comfort_level_strength(occupied_strength):
    if occupied_strength < 150:
        return "Комфортно"
    elif 150 <= occupied_strength <= 275:
        return "Жити можна"
    return "Некомфортно"

def get_comfort_level_total(occupied_cardio, occupied_strength):
    total_occupied = occupied_cardio + occupied_strength
    if total_occupied <= 200:
        return "Комфортно"
    elif 200 < total_occupied <= 325:
        return "Тяжко, але можна"
    return "Некомфортно"

def validate_time_format(time_str):
    try:
        hour, minute = map(int, time_str.split('.'))
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            print("Помилка: година повинна бути в діапазоні 0-23, а хвилина - 0-59.")
            return False
        return True
    except ValueError:
        print("Помилка: введіть час у форматі год:хв, наприклад, 10.30.")
        return False

def is_within_operating_hours(hour):
    if 8 <= hour <= 22:
        return True
    else:
        print("Помилка: час повинен бути в межах робочих годин залу (з 8:00 до 22:00).")
        return False

def predict_comfort(arrival_time, is_cardio_input):
    if not validate_time_format(arrival_time):
        return
    
    hour = int(arrival_time.split('.')[0])
    if not is_within_operating_hours(hour):
        return
    
    is_cardio = 1 if is_cardio_input.lower() == 'так' else 0
    user_input = np.array([[time_to_minutes(arrival_time), is_cardio]])
    user_input_scaled = scaler.transform(user_input)
    
    comfort = model.predict(user_input_scaled)[0][0]

    visitor_count = get_visitor_count_at_time(hour)  
    
    occupied_cardio, occupied_strength = predict_occupied_trainers(hour, visitor_count)
    
    comfort_cardio = get_comfort_level_cardio(occupied_cardio)
    
    comfort_strength = get_comfort_level_strength(occupied_strength)
    
    comfort_total = get_comfort_level_total(occupied_cardio, occupied_strength)
    
    print(f"Кількість відвідувачів у залі на час {arrival_time}: {visitor_count}")
    print(f"Зайнято кардіо-тренажерів: {occupied_cardio} з {CARDIO_TRAINERS} - {comfort_cardio}")
    print(f"Зайнято силових тренажерів: {occupied_strength} з {STRENGTH_TRAINERS} - {comfort_strength}")
    print(f"Загальна кількість зайнятих тренажерів: {occupied_cardio + occupied_strength} з {TOTAL_TRAINERS} - {comfort_total}")
    print(f"Прогнозоване значення комфортності (шкала від 0 до 100): {comfort:.2f}")
    print(f"Загальна оцінка комфортності: {comfort_total}")

arrival_time_input = input("Введіть час приходу (формат год:хв, наприклад, 10.30): ")
is_cardio_input = input("Чи робите ви кардіо? (так/ні): ")

predict_comfort(arrival_time_input, is_cardio_input)
