import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Завантажуємо дані з обох файлів
file_path_visitors = r'C:\Users\Home\Documents\GitHub\KPRS_RGR\gum_visitors.csv'
file_path_visitors_time = r'C:\Users\Home\Documents\GitHub\KPRS_RGR\gum_visitors_time.csv'

data_visitors = pd.read_csv(file_path_visitors)
data_visitors_time = pd.read_csv(file_path_visitors_time)

# Перевірка структури даних
print("Структура даних для gum_visitors:")
print(data_visitors.head())  # Переглядаємо перші кілька рядків з gum_visitors

print("Структура даних для gum_visitors_time:")
print(data_visitors_time.head())  # Переглядаємо перші кілька рядків з gum_visitors_time

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

# Функція для підрахунку кількості відвідувачів по часу з gum_visitors_time
def get_visitor_count_at_time(hour):
    # Перевіряємо кількість відвідувачів у таблиці gum_visitors_time на дану годину
    visitor_count = data_visitors_time[data_visitors_time['Година'] == hour]['Кількість відвідувачів'].values
    return visitor_count[0] if len(visitor_count) > 0 else 0

# Функція для визначення, скільки тренажерів буде зайнято в цей час
def get_occupied_trainers(arrival_time, is_cardio, existing_visitors):
    arrival_minutes = time_to_minutes(arrival_time)
    
    # Підраховуємо зайняті кардіо-тренажери
    occupied_cardio = sum(1 for _, visitor in existing_visitors.iterrows() 
                          if visitor['Кардіо'] == 'Так' and 
                             time_to_minutes(visitor['Час прийшли']) < arrival_minutes < time_to_minutes(visitor['Час вийшли']))
    
    # Підраховуємо зайняті силові тренажери
    occupied_strength = sum(1 for _, visitor in existing_visitors.iterrows() 
                            if visitor['Кардіо'] == 'Ні' and 
                               time_to_minutes(visitor['Час прийшли']) < arrival_minutes < time_to_minutes(visitor['Час вийшли']))
    
    return occupied_cardio, occupied_strength

# Підготовка даних для навчання
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

def predict_occupied_trainers(hour, total_visitors, cardio_ratio=0.25):
    """
    Функція прогнозує кількість зайнятих тренажерів на основі кількості відвідувачів і співвідношення кардіо/сили.
    - hour: година, на яку потрібно зробити прогноз
    - total_visitors: загальна кількість відвідувачів на цей час
    - cardio_ratio: співвідношення відвідувачів, які займаються кардіо (за замовчуванням 25%)
    """
    cardio_visitors = int(total_visitors * cardio_ratio)  # Загальна кількість відвідувачів на кардіо
    strength_visitors = total_visitors - cardio_visitors  # Всі інші займаються силовими тренажерами
    
    # Якщо людей більше, ніж тренажерів, то кожен тренажер буде "завантажений" на 100%
    occupied_cardio = min(cardio_visitors, CARDIO_TRAINERS)
    occupied_strength = min(strength_visitors, STRENGTH_TRAINERS)
    
    # Обчислення того, скільки тренажерів буде зайнято пропорційно відвідувачам
    extra_visitors = total_visitors - (CARDIO_TRAINERS + STRENGTH_TRAINERS)
    if extra_visitors > 0:
        extra_cardio = int(extra_visitors * cardio_ratio)
        extra_strength = extra_visitors - extra_cardio
        occupied_cardio += extra_cardio
        occupied_strength += extra_strength

    return occupied_cardio, occupied_strength

def get_comfort_level_cardio(occupied_cardio):
    """Оцінка комфортності для кардіо тренажерів."""
    if occupied_cardio > 50:
        return "Некомфортно"
    return "Комфортно"

def get_comfort_level_strength(occupied_strength):
    """Оцінка комфортності для силових тренажерів."""
    if occupied_strength < 150:
        return "Комфортно"
    elif 150 <= occupied_strength <= 275:
        return "Терпимо"
    return "Некомфортно"

def get_comfort_level_total(occupied_cardio, occupied_strength):
    """Оцінка загальної комфортності тренажерів."""
    total_occupied = occupied_cardio + occupied_strength
    if total_occupied <= 200:
        return "Комфортно"
    elif 200 < total_occupied <= 325:
        return "Тяжко, але можна"
    return "Некомфортно"

def predict_comfort(arrival_time, is_cardio_input):
    is_cardio = 1 if is_cardio_input.lower() == 'так' else 0
    user_input = np.array([[time_to_minutes(arrival_time), is_cardio]])
    user_input_scaled = scaler.transform(user_input)
    
    # Прогнозуємо комфортність з використанням нейромережі
    comfort = model.predict(user_input_scaled)[0][0]

    # Підраховуємо кількість відвідувачів з gum_visitors_time
    hour = int(arrival_time.split('.')[0])
    visitor_count = get_visitor_count_at_time(hour)  # Отримуємо кількість відвідувачів для цієї години
    
    # Прогнозуємо кількість зайнятих тренажерів на основі кількості відвідувачів
    occupied_cardio, occupied_strength = predict_occupied_trainers(hour, visitor_count)
    
    # Оцінка для кардіо
    comfort_cardio = get_comfort_level_cardio(occupied_cardio)
    
    # Оцінка для силових тренажерів
    comfort_strength = get_comfort_level_strength(occupied_strength)
    
    # Оцінка для загальної кількості тренажерів
    comfort_total = get_comfort_level_total(occupied_cardio, occupied_strength)
    
    # Загальна комфортність, яка бере до уваги нейромережу
    if comfort >= 0.5:
        neural_net_comfort = "Комфортно"
    else:
        neural_net_comfort = "Не комфортно"
    
    # Виведення результатів
    print(f"Кількість відвідувачів у залі на час {arrival_time}: {visitor_count}")
    print(f"Зайнято кардіо-тренажерів: {occupied_cardio} з {CARDIO_TRAINERS} - {comfort_cardio}")
    print(f"Зайнято силових тренажерів: {occupied_strength} з {STRENGTH_TRAINERS} - {comfort_strength}")
    print(f"Загальна кількість зайнятих тренажерів: {occupied_cardio + occupied_strength} з {TOTAL_TRAINERS} - {comfort_total}")
    print(f"Шкала комфортності (від нейромережі): {neural_net_comfort}")
    print(f"Загальна оцінка комфортності: {comfort_total} (з нейромережею: {neural_net_comfort})")

# Запит користувача
arrival_time_input = input("Введіть час приходу (формат год:хв, наприклад, 10.30): ")
is_cardio_input = input("Чи робите ви кардіо? (так/ні): ")

# Прогнозування комфортності
predict_comfort(arrival_time_input, is_cardio_input)