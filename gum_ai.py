import pandas as pd
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Завантажуємо дані з файлу
file_path = r'C:\Users\Home\Documents\GitHub\KPRS_RGR\gum_visitors.csv'
data = pd.read_csv(file_path)

# Константи
TOTAL_TRAINERS = 200
CARDIO_TRAINERS = 50
STRENGTH_TRAINERS = 150
CARDIO_DURATION = 20  # хвилин

# Функція для переведення часу в хвилини
def time_to_minutes(time_str):
    if isinstance(time_str, str):
        hour, minute = map(int, time_str.split('.'))
        return hour * 60 + minute
    else:
        return 0  # Якщо значення не рядок, повертаємо 0

# Функція для визначення, чи буде комфортно тренуватися
def is_comfortable(arrival_time, is_cardio, existing_visitors):
    arrival_minutes = time_to_minutes(arrival_time)
    
    # Визначаємо, скільки тренажерів буде зайнято в цей час
    occupied_cardio = sum(1 for _, visitor in existing_visitors.iterrows() 
                          if visitor['Кардіо'] == 'Так' and time_to_minutes(visitor['Час прийшли']) <= arrival_minutes < time_to_minutes(visitor['Час вийшли']))
    occupied_strength = sum(1 for _, visitor in existing_visitors.iterrows() 
                            if visitor['Кардіо'] == 'Ні' and time_to_minutes(visitor['Час прийшли']) <= arrival_minutes < time_to_minutes(visitor['Час вийшли']))
    
    if is_cardio:
        if occupied_cardio < CARDIO_TRAINERS:
            return True
    else:
        if occupied_strength < STRENGTH_TRAINERS:
            return True
    
    return False

# Підготовка даних для навчання
X = []
y = []

data['Час прийшли хвилини'] = data['Час прийшли'].apply(time_to_minutes)
data['Час вийшли хвилини'] = data['Час вийшли'].apply(time_to_minutes)

for _, row in data.iterrows():
    arrival_time = row['Час прийшли']
    is_cardio = 1 if row['Кардіо'] == 'Так' else 0
    departure_time = row['Час вийшли']
    existing_visitors = data[(data['Час прийшли хвилини'] <= arrival_time) & (data['Час вийшли хвилини'] > arrival_time)]
    
    comfortable = is_comfortable(arrival_time, is_cardio, existing_visitors)
    
    X.append([arrival_time, is_cardio])
    y.append(comfortable)

# Стандартизуємо дані
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Переконаємось, що X_scaled має правильну форму
X_scaled = np.array(X_scaled)

# Розділяємо на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Створення нейронної мережі
model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Вихід: 0 - не комфортно, 1 - комфортно

# Компіляція моделі
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Навчання моделі
model.fit(X_train, np.array(y_train), epochs=50, batch_size=16, validation_data=(X_test, np.array(y_test)))

# Функція для запиту користувача та прогнозування
def predict_comfort(arrival_time, is_cardio_input):
    is_cardio = 1 if is_cardio_input.lower() == 'так' else 0
    user_input = np.array([[time_to_minutes(arrival_time), is_cardio]])
    user_input_scaled = scaler.transform(user_input)
    
    comfort = model.predict(user_input_scaled)[0][0]
    return "Комфортно" if comfort >= 0.5 else "Не комфортно"

# Запит користувача
arrival_time_input = input("Введіть час приходу (формат год:хв, наприклад, 10.30): ")
is_cardio_input = input("Чи робите ви кардіо? (так/ні): ")

# Прогнозування комфортності
result = predict_comfort(arrival_time_input, is_cardio_input)
print(result)
