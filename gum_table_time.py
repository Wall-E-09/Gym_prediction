import pandas as pd
import numpy as np

file_path = r'C:\Users\Home\Documents\GitHub\KPRS_RGR\gum_visitors.csv'
data = pd.read_csv(file_path)

print(data.dtypes)

data['Час прийшли година'] = data['Час прийшли']
data['Час вийшли година'] = data['Час вийшли']

visitor_counts = []

for hour in range(24):
    visitors_in_hour = sum(1 for _, visitor in data.iterrows() 
                           if visitor['Час прийшли година'] <= hour < visitor['Час вийшли година'])
    visitor_counts.append([hour, visitors_in_hour])

visitor_df = pd.DataFrame(visitor_counts, columns=['Година', 'Кількість відвідувачів'])

visitor_df.to_csv(r'C:\Users\Home\Documents\GitHub\KPRS_RGR\gum_visitors_time.csv', index=False)

print(visitor_df)
