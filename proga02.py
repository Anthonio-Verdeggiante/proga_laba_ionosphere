import numpy as np

from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt


# Читаем, базовая процедура
file_path = '10.02.25 01'  
data = []

with open(file_path, 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        timestamp = parts[0].split()
        date, time = timestamp
        year, month, day = map(int, date.split('.'))
        hours, minutes, seconds = map(int, time.split(':'))
        
        
        amplitude = float(parts[1] + parts[2].strip())
        if seconds <= 40:  
            data.append([seconds, amplitude])
        else:
            break  

# Преобразование данных в DataFrame для удобства
df = pd.DataFrame(data, columns=['Seconds', 'Amplitude'])

# Тут происходит следующее: такт записи амплитуды - переменная, зависящая от номера секунды
# Т.е. на 11-й секунде было выполнено 8 отсчетов, на 12-й уже 9 и так далее кароч
# Я тут пытаюсь реализовать выборку по крайнему значению амплитуды каждой конкретной секунды
# Иными словами, беру последнее значение амплитуды замера i-й секунды с 
# учетом приращения такта на единицу следующей, (i+1)-й секунды
df['Next_Second'] = df['Seconds'].shift(-1)
df = df[df['Next_Second'] != df['Seconds']]

# Создание массивов
seconds = df['Seconds'].values
amplitudes = df['Amplitude'].values
amplitudes0 = []

# Буду иногда выводить массивы в терминал для контроля правильности
print("Seconds:", seconds)
print("Amplitudes:", amplitudes)




# Значение амплитуды в файле - дробное число. после горе-парсинга на заполнение в массив
# поступают значения после запятой. Так что тут кароч я собираю значения в новый массив,
# учитывающий смещение в показателе десятки на -6, что соответствует истинным значениям амплитуд
for i in range(len(amplitudes)):
    amplitudes0.append(np.zeros)

for i in range(len(amplitudes)):
    amplitudes0 = amplitudes * 10**(-6)

print(amplitudes0)

# Здесь я пытаюсь аппроксимировать библиотечным полиномом, это обяз пригодится в расчёте СПМ
tck = interpolate.splrep(seconds, amplitudes0, s=0)
t = np.linspace(min(seconds), max(seconds), 300)  
f = interpolate.splev(t, tck, der=0)



# Построение графика сигнала по времени
plt.figure(figsize=(10, 5))
plt.plot(seconds, amplitudes0, 'bo-', label='Исходные данные')
plt.plot(t, f, 'r-', label='Аппроксимация')
plt.title('Зависимость амплитуды от времени')
plt.xlabel('t, c')
plt.ylabel('Аmpl, V')
plt.grid(True)
plt.show()


# РАСЧЁТ СПМ

N = len(amplitudes0)
delta_omeha = 0.1

# Делаю дискретный Фурье-образ сигнала по частотам
# Частоты одностороннего спектра, т.к. из-за модуляции может вылезти гармоника
# по верхней боковой полосе
ff = np.fft.fft(f)
tf = np.fft.fftfreq(N, delta_omeha)[:N//2]


S = 2.0/N * np.abs(ff[:N//2])**2

print(S)

plt.figure(figsize=(10, 6))
plt.semilogy(tf, S)
plt.title('Спектральная плотность мощности')
plt.xlabel('Частота (Гц)')
plt.ylabel('СПМ')
plt.grid(True)
plt.show()


# РАСЧЁТ КОЭФФИЦИЕНТА АВТОКОРРЕЛЯЦИИ

# Вощм как мы помним по статам, корреляцию СПМ можно рассчитать через формулу Винера-Хинчина
# Она представляет собой обратое Фурье от СПМ, отображаем его зеркально
spm_sym = np.concatenate([S, S[1:-1][::-1]])

# Здесь беру только реальную часть, т.к. коэф. автокорра не является комплекснозначным
autocorr = np.fft.ifft(spm_sym).real  

# Нормировку по единице опа сюда
autocorr /= autocorr[0]

# Поскольку автокорреляция симметрична, покажем только половину
lags = np.arange(0, N-2)
plt.figure(figsize=(12, 6))
plt.plot(lags, autocorr[:N])
plt.title('Функция автокорреляции через СПМ')
plt.xlabel('Лаги')
plt.ylabel('Коэффициент автокорреляции')
plt.grid(True)
plt.show()
