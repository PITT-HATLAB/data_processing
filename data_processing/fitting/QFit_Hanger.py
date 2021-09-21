import numpy as np
import matplotlib.pyplot as plt
import inspect
from scipy.optimize import curve_fit
import easygui
from plottr.data.datadict_storage import all_datadicts_from_hdf5

FREQ_UNIT = {'GHz' : 1e9,
             'MHz' : 1e6,
             'KHz' : 1e3,
             'Hz' : 1.0
             }


def rounder(value):
    return "{:.4e}".format(value)



def hangerFuncMagAndPhase(freq, Qext, Qint, f0, magBack, delta, phaseCorrect):
    omega0=f0
    
    x = (freq - omega0)/(omega0)
    S_21_up = Qext + 1j * Qext * Qint * (2 * x + 2 * delta / omega0)
    S_21_down = (Qint + Qext) + 2 * 1j * Qext * Qint * x

    S21 = magBack * (S_21_up / S_21_down) * np.exp(1j * (phaseCorrect)) #model by Kurtis Geerlings thesis
    
    mag = np.log10(np.abs(S21)) * 20
    fase = np.angle(S21)
    
    return (mag + 1j * fase).view(float)


def getData_from_datadict(filepath, plot_data = None): 
    datadict = all_datadicts_from_hdf5(filepath)['data']
    powers_dB = datadict.extract('power')['power']['values']
    freqs = datadict.extract('power')['frequency']['values']*2*np.pi
    phase_rad = datadict.extract('phase')['phase']['values']
    
    lin = np.power(10, powers_dB/20)
    real = lin*np.cos(phase_rad)
    imag = lin*np.sin(phase_rad)
    
    test = 0 #if set to 1, can be used to plot real data
    if test:
        plt.figure('dataPlot', figsize=(9,9))
        plt.subplot(221)
        plt.title('mag (db)')
        plt.plot(freqs/(2*np.pi), powers_dB)
        plt.subplot(222)
        plt.title('phase (rad)')
        plt.plot(freqs/(2*np.pi), phase_rad)
        plt.subplot(223)
        plt.title('real')
        plt.plot(freqs/(2*np.pi), real)
        plt.subplot(224)
        plt.title('imag')
        plt.plot(freqs/(2*np.pi), imag)
    
    if plot_data:
        plt.figure('mag')
        plt.plot(freqs, powers_dB)
        plt.figure('phase')
        plt.plot(freqs, phase_rad)
        
    return (freqs, real, imag, powers_dB, phase_rad)
            

def fit(freq, real, imag, mag, phase, Qguess=(2e5, 1e5), bounds = None, f0Guess = None, magBackGuess = None, delta = 1e6, phaseGuess = 0.3*np.pi):
    if f0Guess == None:
         f0Guess = freq[int(np.floor(np.size(freq)/2))] #dumb guess of "it's probably in the middle"
        # #smart guess of "it's probably the lowest point"
        # f0Guess = 14.989e9*2*np.pi
    print("Guess freq: "+str(f0Guess/(2*np.pi*1e9)))
    # if phaseSlopeGuess == None:
    #     phaseSlopeGuess = (phase[9]-phase[2])/(freq[9]-freq[2]) #Slope of the phase within the first few points of data
    # print("phaseSlopeGuess: "+str(phaseSlopeGuess))
    lin = 10**(mag / 20.0)
    if magBackGuess == None: 
        magBackGuess = np.average(lin[:int(len(freq) / 5)])
    QextGuess = Qguess[0]
    QintGuess = Qguess[1]
    if bounds == None: 
        bounds=([QextGuess / 5, QintGuess / 5, f0Guess/1.00001, magBackGuess / 10, -1e9, phaseGuess*1.1], #bounds can be made tighter to allow better convergences
                [QextGuess * 5, QintGuess * 5, f0Guess*1.00001, magBackGuess * 10, 1e9, phaseGuess/1.1])
    
    target_func = hangerFuncMagAndPhase
    data_to_fit = (mag  + 1j * phase).view(float)
    # data_to_fit = (real  + 1j * imag).view(float)
    popt, pcov = curve_fit(target_func, freq, data_to_fit, 
                            p0=(QextGuess, QintGuess, f0Guess, magBackGuess, delta, phaseGuess),
                            bounds=bounds,
                            maxfev=1e4, ftol=2.3e-16, xtol=2.3e-16)

    return popt, pcov


def plotRes(freq, real, imag, mag, phase, popt):
    xdata = freq/ (2 * np.pi)
    magRes = hangerFuncMagAndPhase(freq, *popt)[::2]
    faseRes = hangerFuncMagAndPhase(freq, *popt)[1::2]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title('mag (db)')
    plt.plot(xdata, mag, '.')
    plt.plot(xdata, magRes)
    plt.subplot(1, 2, 2)
    plt.title('phase')
    plt.plot(xdata, phase, '.')
    plt.plot(xdata, faseRes)
    plt.show()





if __name__ == '__main__':
    # filepath = easygui.fileopenbox()
    filepath = r'Z:/Data/HuntCollab2/hBN_cap/20210917_cooldown/2021-09-21/2021-09-21_0004_11_28ghz_mode_50mhzspan/2021-09-21_0004_11_28ghz_mode_50mhzspan.ddh5'
    (freq, real, imag, mag, phase) = getData_from_datadict(filepath, plot_data=0)
    ltrim = 350 #trims the data if needed
    rtrim = 400 #keep this value greater than 1
    freq = freq[ltrim:-rtrim]
    real = real[ltrim:-rtrim]
    imag = imag[ltrim:-rtrim]
    mag = mag[ltrim:-rtrim]
    phase = phase[ltrim:-rtrim]
    

    popt, pcov = fit(freq, real, imag, mag, phase, Qguess=(1e4, 9e2), magBackGuess = 1e-2, f0Guess = 1.12839e10*2*np.pi, delta = 1e6, phaseGuess = -0.45*np.pi)  #(ext, int)   
    print(f'f (Hz): {rounder(popt[2]/2/np.pi)}', )
    fitting_params = list(inspect.signature(hangerFuncMagAndPhase).parameters.keys())[1:]
    for i in range(2):
        print(f'{fitting_params[i]}: {rounder(popt[i])} +- {rounder(np.sqrt(pcov[i, i]))}')
    Qtot = popt[0] * popt[1] / (popt[0] + popt[1])
    print('Q_tot: ', rounder(Qtot), '\nT1 (s):', rounder(Qtot/popt[2]), f"Kappa: {rounder(popt[2]/2/np.pi/Qtot)}" )
    print("Magback val: ", popt[3])
    print("delta val: ", popt[4])
    print("phaseCorrect val: ", popt[5])


    

    plotRes(freq, real, imag, mag, phase, popt)
