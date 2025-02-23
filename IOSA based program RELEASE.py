import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt

# Чтение и парсинг
file_path = '19.02.2024_12h30m.2024_12h30m'
data = []
prev_seconds = -1
time_offset = 0
try:
    with open(file_path, 'r') as file:
        for i in range(940):
            next(file)
        for line in file:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            timestamp = parts[0].split()
            date, time = timestamp
            year, month, day = map(int, date.split('.'))
            hours, minutes, seconds = map(int, time.split(':'))
            amplitude = float(parts[1] + parts[2].strip())
            if seconds < prev_seconds:
                time_offset += 60
            corrected_seconds = seconds + time_offset
            if corrected_seconds <= 150:
                data.append([corrected_seconds, amplitude])
            else:
                break
            prev_seconds = seconds
except FileNotFoundError:
    print(f"Файл {file_path} не найден")
    exit()
except ValueError as e:
    print(f"Ошибка парсинга данных: {e}")
    exit()

if not data:
    print("Нет данных для обработки")
    exit()

# Преобразование и фильтрация
df = pd.DataFrame(data, columns=['Seconds', 'Amplitude'])
df['Next_Second'] = df['Seconds'].shift(-1)
df = df[df['Next_Second'] != df['Seconds']]
seconds = df['Seconds'].values
amplitudes = df['Amplitude'].values

# Коррекция амплитуд
amplitudes0 = amplitudes * 10**(-6)
print("Seconds:", seconds)
print("Amplitudes0:", amplitudes0)

# Интерполяция
if len(seconds) < 2:
    print("Недостаточно данных для интерполяции")
    exit()
tck = interpolate.splrep(seconds, amplitudes0, s=0)
t = np.linspace(min(seconds), max(seconds), 300)
f = interpolate.splev(t, tck, der=0)

# График сигнала
plt.figure(figsize=(10, 5))
plt.plot(seconds, amplitudes0, 'bo-', label='Исходные данные')
plt.plot(t, f, 'r-', label='Аппроксимация')
plt.title('Зависимость амплитуды от времени')
plt.xlabel('t, c')
plt.ylabel('Аmpl, V')
plt.grid(True)
plt.legend()
plt.show()

# СПМ
N = len(f)
delta_omega = 0.1
ff = np.fft.fft(f)
tf = np.fft.fftfreq(N, delta_omega)[:N//2]
S = 2.0 / N * np.abs(ff[:N//2])**2
plt.figure(figsize=(10, 6))
plt.semilogy(tf, S)
plt.title('Спектральная плотность мощности')
plt.xlabel('Частота (Гц)')
plt.ylabel('СПМ')
plt.grid(True)
plt.show()

# Автокорреляция
spm_sym = np.concatenate([S, S[1:-1][::-1]])
autocorr = np.fft.ifft(spm_sym).real
autocorr /= autocorr[0]
lags = np.arange(0, len(autocorr))
plt.figure(figsize=(12, 6))
plt.plot(lags, autocorr)
plt.title('Функция автокорреляции через СПМ')
plt.xlabel('Время, с')
plt.ylabel('Коэффициент автокорреляции')
plt.grid(True)
plt.show()