import numpy as np
from polarData import polarData
from pythonAudioMeasurements.audioSample import audioSample
from pythonAudioMeasurements.microphone import Microphone
from pythonAudioMeasurements.microphoneArray import MicrophoneArray
import matplotlib.pyplot as plt
from scipy.signal import convolve


def test_mic_apply():
    filename = "/home/terrasa/UROP/polar-measurement/data/19_Jan15/spv1840.pkl" 
    pd = polarData.fromPkl(filename)

    f_targ = 150

    pd.plotAngle(90)

    # position in mm
    mic = Microphone(pd, [-1,1])

    fs = 44.1e3
    n = np.arange(4000)

    sin_wave = np.sin(n*(2*np.pi)*(f_targ/fs))
    sin_wave = audioSample(sin_wave, type="t", Fs=fs)

    # plot time-domain wave-form
    sin_wave.toTime()
    plt.plot(sin_wave.t(), sin_wave)


    sin_shifted = mic.apply_microphone(sin_wave, 90)

    # plot the resulting waveform
    sin_shifted.toTime()
    plt.plot(sin_shifted.t(), sin_shifted)

    plt.legend(["original", "shifted"])
    plt.show()

def test_xy():
    filename = "/home/terrasa/UROP/polar-measurement/data/19_Jan15/spv1840.pkl" 
    pd = polarData.fromPkl(filename)

    f_targ = 150

    pd.plotAngle(90)

    # position in mm
    mic = Microphone(pd, [-500,2000])

    fs = 44.1e3
    n = np.arange(10000)

    sin_wave = np.sin(n*(2*np.pi)*(f_targ/fs))
    sin_wave = audioSample(sin_wave, type="t", Fs=fs)

    # plot time-domain wave-form
    sin_wave.toTime()
    plt.plot(sin_wave.t(), sin_wave)

    sin_shifted = mic.apply(sin_wave, 90)

    # plot the resulting waveform
    sin_shifted.toTime()
    plt.plot(sin_shifted.t(), sin_shifted)

    plt.legend(["original", "shifted"])
    plt.show()


def simulate_polar_1mic():

    filename = "/home/terrasa/UROP/polar-measurement/data/19_Jan15/spv1840.pkl" 
    pd = polarData.fromPkl(filename)


    f_options = np.logspace(0,4, 6)

    pd.plotAngle(90)

    # position in mm
    mic = Microphone(pd, [-500,2000])

    fs = 44.1e3
    n = np.arange(10000)

    for f_test in f_options:

        sin_wave = np.sin(n*(2*np.pi)*(f_test/fs))
        sin_wave = audioSample(sin_wave, type="t", Fs=fs)


        thetas = np.array(list(range(0, 361, 10)))
        mags = []

        for theta in thetas:

            result = mic.apply(sin_wave, theta)
            result.toTime()
            mags.append(max(result))

        plt.polar(thetas*np.pi/180, mags)

    plt.legend(f_options)
    plt.show()


def simulate_polar_2mic():

    filename = "/home/terrasa/UROP/polar-measurement/data/19_Jan15/spv1840.pkl" 
    pd = polarData.fromPkl(filename)


    f_options = [440]

    # pd.plotAngle(90)

    # position in mm
    mic_1 = Microphone(pd, [0,0])
    mic_2 = Microphone(pd, [390, 0])

    mic_array = MicrophoneArray([mic_1, mic_2])

    fs = 44.1e3
    n = np.arange(10001)

    for f_test in f_options:

        sin_wave = np.sin(n*(2*np.pi)*(f_test/fs))
        sin_wave = audioSample(sin_wave, type="t", Fs=fs)


        thetas = np.array(list(range(0, 361, 1)))
        mags = []

        for theta in thetas:

            result = mic_array.apply(sin_wave, theta)
            result.toTime()

            # result.plot(True)
            mags.append(max(result))

        plt.polar(thetas*np.pi/180, mags)

    plt.legend(f_options)
    plt.show()




if __name__ == "__main__":
    # test_mic_apply()
    # test_xy()
    simulate_polar_2mic()