import pandas as pd
import numpy as np

# Завантажуємо дані з файлу
file_path = r'C:\Users\Home\Documents\GitHub\KPRS_RGR\gum_visitors.csv'
data = pd.read_csv(file_path)

# Перевіримо типи даних у стовпцях 'Час прийшли' та 'Час вийшли'
print(data.dtypes)

# Якщо час у форматі float (десяткові числа), використовуємо його без змін:
data['Час прийшли година'] = data['Час прийшли']
data['Час вийшли година'] = data['Час вийшли']

# Створюємо нову таблицю для підрахунку відвідувачів по годинах
visitor_counts = []

# Перевіримо години з 0 до 23
for hour in range(24):
    visitors_in_hour = sum(1 for _, visitor in data.iterrows() 
                           if visitor['Час прийшли година'] <= hour < visitor['Час вийшли година'])
    visitor_counts.append([hour, visitors_in_hour])

# Створюємо нову таблицю з кількістю відвідувачів по годинах
visitor_df = pd.DataFrame(visitor_counts, columns=['Година', 'Кількість відвідувачів'])

# Зберігаємо таблицю в CSV файл
visitor_df.to_csv(r'C:\Users\Home\Documents\GitHub\KPRS_RGR\gum_visitors_time.csv', index=False)

# Виводимо таблицю
print(visitor_df)
