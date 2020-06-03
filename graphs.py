import matplotlib.pyplot as plt
import csv
import pandas as pd

x = []
y = []

"""with open('testing.csv','r') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    for row in data:
        x.append(float(row[0]))
        y.append(float(row[1]))"""

df = pd.read_csv('manipal_weather.csv')
print(df.head())

plt.plot(df['time'].values,df['temperatureHigh'].values, label='Loaded from file!')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Max Temperature')
plt.legend()
plt.show()