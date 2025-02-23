import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics import tsaplots
from scipy.fft import fft, fftfreq
import numpy as np

f = open('10.02.25 01', 'r')
actual = 0
preActual = 0
data = f.readlines()
i = 0
print(data[0].split())
print(data[1].split())
print(data[2].split())
amplitude = float(data[0].split()[2][:-1].replace(',', '.'))
print(amplitude)
amplitudes = []
rowCount = 0
size = 150

for i in range(len(data) - 1):
    actual = int(data[i].split()[3])
    # amplitudes.append(float(data[i].split()[2][:-1].replace(',', '.')))
    if actual != preActual:
        row = []
        amplitudes.append(row)
        for j in range(i, i + size):
            amplitudes[rowCount].append(float(data[j].split()[2][:-1].replace(',', '.')))
        preActual = actual
        i += 30
        rowCount += 1
    i += 1

sampleRate = 10
duration = 15

def furie(zamir):
    spm = []
    acf = sm.tsa.acf(zamir, nlags=size)
    stepsFreq = 1000
    for m in range(stepsFreq):
        s = 0
        for n in range(len(zamir)):
            b = (1 / stepsFreq )* 0.1 * n * m
            s = s + acf[n] * np.exp(b*1j)
        spm.append(s)
    return spm


def furiefreq():
    frqs = []
    for i in range(1000):
        frqs.append(i/1000)
    return frqs

def pictures(zamir):
    plt.figure()
    plt.plot(zamir)
    x = pd.plotting.autocorrelation_plot(amplitudes[0])
    x.plot()
    fig = tsaplots.plot_acf(amplitudes[0], lags = 50)
    acf = sm.tsa.acf(zamir, nlags=size)
    plt.figure()
    plt.plot(sm.tsa.acf(zamir, nlags=size))
    plt.figure()
    N = sampleRate * duration
    y = furie(zamir)
    print(y)
    xf = furiefreq()
    plt.plot(xf, np.abs(y))



for i in range(len(amplitudes)):
    pictures(amplitudes[i])

acf = sm.tsa.acf(amplitudes[0], nlags=size)
z = 3 + 4j
b = 5 + 4
result = np.exp((5+4)*1j)
print(np.abs(result))
plt.show()

