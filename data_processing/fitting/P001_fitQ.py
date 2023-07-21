import numpy as np
import matplotlib.pyplot as plt
import csv
import h5py
import inspect
from scipy.optimize import curve_fit

# import easygui

FREQ_UNIT = {'GHz': 1e9,
             'MHz': 1e6,
             'KHz': 1e3,
             'Hz': 1.0
             }


def rounder(value, dig=4):
    return f"{value:.{dig}e}"


def reflectionFunc(freq, Qext, Qint, f0, magBack, phaseCorrect):
    omega0 = f0
    delta = freq - omega0
    S_11_up = 1.0 / (1j * delta * (2 + delta / omega0) / (1 + delta / omega0) + omega0 / Qint) - Qext / omega0
    S_11_down = 1.0 / (1j * delta * (2 + delta / omega0) / (1 + delta / omega0) + omega0 / Qint) + Qext / omega0
    S11 = magBack * (S_11_up / S_11_down) * np.exp(1j * (phaseCorrect))
    realPart = np.real(S11)
    imagPart = np.imag(S11)

    return (realPart + 1j * imagPart).view(float)
    # return realPart 
    # return imagPart 


def reflectionFunc_re(freq, Qext, Qint, f0, magBack, phaseCorrect):
    return reflectionFunc(freq, Qext, Qint, f0, magBack, phaseCorrect)[::2]


def getData(filename, method='vna_old', freq_unit='GHz', plot_data=1):
    if method == 'hfss':
        """The csv file must be inthe format of:
            freq  mag(dB)  phase(cang_deg)  
        """
        with open(filename) as csvfile:
            csvData = list(csv.reader(csvfile))
            csvData.pop(0)  # Remove the header
            data = np.zeros((len(csvData[0]), len(csvData)))
            for x in range(len(csvData)):
                for y in range(len(csvData[0])):
                    data[y][x] = csvData[x][y]

        freq = data[0] * 2 * np.pi * FREQ_UNIT[freq_unit]  # omega
        phase = np.array(data[2]) / 180. * np.pi
        mag = data[1]
        lin = 10 ** (mag / 20.0)

    elif method == 'vna':
        f = h5py.File(filename, 'r')
        freq = f['VNA Frequency (Hz)'][()] * 2 * np.pi
        phase = f['Phase (deg)'][()] / 180. * np.pi
        mag = f['Power (dB)'][()]
        lin = 10 ** (mag / 20.0)
        f.close()
        
    elif method == 'vna_2':
        f = h5py.File(filename, 'r')
        freq = f['data']['frequency'][()] * 2 * np.pi
        phase = f['data']['phase'][()]
        mag = f['data']['power'][()]
        lin = 10 ** (mag / 20.0)
        f.close()

    elif method == 'vna_old':
        f = h5py.File(filename, 'r')
        freq = f['Freq'][()] * 2 * np.pi
        phase = f['S21'][()][1] / 180 * np.pi
        mag = f['S21'][()][0]
        lin = 10 ** (mag / 20.0)
        f.close()

    else:
        raise NotImplementedError('method not supported')

    real = lin * np.cos(phase)
    imag = lin * np.sin(phase)

    if plot_data:
        plt.figure(figsize=(12,5))
        plt.subplot(1, 2, 1)
        plt.title('mag')
        plt.plot(freq / 2 / np.pi, mag)
        plt.subplot(1, 2, 2)
        plt.title('phase')
        plt.plot(freq / 2 / np.pi, phase)

    return (freq, real, imag, mag, phase)

    # if method == 'vna':  
    #     f = h5py.File(filename,'r')
    #     freq = f['VNA Frequency (Hz)'][()]
    #     phase = f['Phase (deg)'][()] / 180. * np.pi
    #     lin = 10**(f['Power (dB)'][()] / 20.0)
    # if method == 'vna_old': 
    #     f = h5py.File(filename,'r')
    #     freq = f['Freq'][()]
    #     phase = f['S21'][()][0] / 180. * np.pi
    #     lin = 10**(f['S21'][()][1] / 20.0)


def fit(omega, real, imag, mag, phase, Qguess=(1e4, 1e5), real_only=0, omegaGuess="min"):
    # W0Guess = 8.5596e9
    if omegaGuess is "mid":
        W0Guess = omega[int(np.floor(np.size(omega) / 2))] #dumb guess of "it's probably in the middle"
    elif omegaGuess is "min":
        W0Guess = omega[np.argmin(mag)]  # smart guess of "it's probably the lowest point"
    elif type(omegaGuess) in [float, int, np.float64, np.int]:
        W0Guess = omegaGuess
    else:
        raise ("frequency guess must be 'min' or 'mid' or a number")

    # print(W0Guess/2/np.pi)
    lin = 10 ** (mag / 20.0)
    magBackGuess = np.average(lin[:int(len(omega) / 5)])
    QextGuess = Qguess[0]
    QintGuess = Qguess[1]
    if type(omegaGuess) in [float, int]:
        bounds=([QextGuess / 100, QintGuess / 100.0, W0Guess - 30e6, magBackGuess / 10.0, -2 * np.pi],
                [QextGuess * 100, QintGuess * 100.0, W0Guess + 30e6, magBackGuess * 10.0, 2 * np.pi])
    else:
        bounds = ([QextGuess / 20, QintGuess / 10, omega[0], magBackGuess / 10.0, -2 * np.pi],
                  [QextGuess * 20, QintGuess * 10, omega[-1], magBackGuess * 10.0, 2 * np.pi])

    target_func = reflectionFunc
    data_to_fit = (real + 1j * imag).view(float)
    if real_only:
        target_func = reflectionFunc_re
        data_to_fit = real
    popt, pcov = curve_fit(target_func, omega, data_to_fit,
                           p0=(QextGuess, QintGuess, W0Guess, magBackGuess, 0),
                           bounds=bounds,
                           maxfev=1e15, ftol=2.3e-16, xtol=2.3e-16)

    return popt, pcov


def plotRes(freq, real, imag, mag, phase, popt):
    xdata = freq / (2 * np.pi)
    realRes = reflectionFunc(freq, *popt)[::2]
    imagRes = reflectionFunc(freq, *popt)[1::2]
    # realRes = reflectionFunc(freq, *popt)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title('real')
    plt.plot(xdata, real, '.')
    plt.plot(xdata, realRes)
    plt.subplot(1, 2, 2)
    plt.title('imag')
    plt.plot(xdata, imag, '.')
    plt.plot(xdata, imagRes)
    plt.show()

def printFitQResult(popt,pcov):
    Qext = popt[0]
    Qint = popt[1]
    Qtot = popt[0] * popt[1] / (popt[0] + popt[1])
    freq_ = popt[2] / 2 / np.pi
    print(f'f (Hz): {rounder(freq_,9)}+-{rounder(np.sqrt(pcov[2,2])/2/np.pi, 9)}')
    print(f'Qext: {rounder(Qext)}+-{rounder(np.sqrt(pcov[0,0]))}')
    print(f'Qint: {rounder(Qint)}+-{rounder(np.sqrt(pcov[1,1]))}')
    print('Q_tot: ', rounder(Qtot))
    print('T1 (s):', rounder(Qtot / freq_ / 2 / np.pi), '\nMaxT1 (s):', rounder(Qint / freq_ / 2 / np.pi))
    print('kappa/2Pi: ', freq_ / Qext / 1e6, 'MHz')


if __name__ == '__main__':
    plt.close('all')

    # filepath = r'L:\Data\Tree2.1\Modes\20220109\\'
    # filepath = r'L:\Data\SNAIL_Module\0106\SM2\CavMode\\'
    # filename = '7.626GHz_bias-2.5mA_-70dBm'
    # filename = 'C3_Bias-2.3mA_-90dBm'
    # filename = 'C2_CavInOut_Bias-0.74mA_-50dBm'

    filepath = r'L:\Data\2D_T1_tester\20220923_cooldown\r2\\'
    # filepath = r'L:\Data\SNAIL_Module\0106\SM2\CavMode\\'
    # filename = '7.626GHz_bias-2.5mA_-70dBm'
    filename = '2dt1tester_reso_-20dBm'
    filepath = filepath + filename

    (freq, real, imag, mag, phase) = getData(filepath, method="vna_old", plot_data=1)
    trim = 1
    freq = freq[trim:-trim]
    real = real[trim:-trim]
    imag = imag[trim:-trim]
    mag = mag[trim:-trim]
    phase = phase[trim:-trim]

    popt, pcov = fit(freq, real, imag, mag, phase, Qguess=(5e3,6e3) ) # (ext, int)
    printFitQResult(popt, pcov)
    plotRes(freq, real, imag, mag, phase, popt)
