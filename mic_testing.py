import numpy as np
from pythonAudioMeasurements.polarData import polarData
from pythonAudioMeasurements.audioSample import audioSample
from pythonAudioMeasurements.microphone import Microphone
from pythonAudioMeasurements.microphoneArray import MicrophoneArray
import matplotlib.pyplot as plt
from scipy.signal import convolve
import tensorflow as tf


filename = "/home/terrasa/UROP/polar-measurement/data/19_Jan15_fixedpkls/spv1840.pkl" 


def test_mic_apply():
    filename = "/home/terrasa/UROP/polar-measurement/data/19_Jan15/spv1840.pkl" 
    pd = polarData.fromPkl(filename)

    pd.plotAngle(90)

    # position in mm
    mic = Microphone(pd, [-500,2000])

    fs = 44.1e3
    length = 100000
    n = np.arange(length)

    f_targ = (fs/length) * 4900 # (2pi/ T)*k
    # f_targ_2 = 2300

    sin_wave_1 = np.sin(n*(2*np.pi)*(f_targ/fs))
    # sin_wave_2 = np.sin(n*(2*np.pi)*(f_targ_2/fs))
    sin_wave = audioSample(sin_wave_1, type="t", Fs=fs)
    # sin_wave.hanning()

    pad = 10000
    # sin_wave.hanning()
    # sin_wave.zeroPadStart(pad)
    # sin_wave.zeroPadEnd(pad)

    sin_wave.plot(both=True)

    sin_shifted = mic.apply_microphone(sin_wave, 90)

    # plot the resulting waveform
    sin_shifted.toTime()

    sin_shifted.plot(both=True)


def test_xy():
    filename = "/home/terrasa/UROP/polar-measurement/data/19_Jan15/spv1840.pkl" 
    pd = polarData.fromPkl(filename)


    pd.plotAngle(90)

    # position in mm
    mic = Microphone(pd, [-500,2000])

    fs = 44.1e3
    length = 100000
    n = np.arange(length)

    f_targ = (fs/length) * 4000 # (2pi/ T)*k
    # f_targ_2 = 2300

    sin_wave_1 = np.sin(n*(2*np.pi)*(f_targ/fs))
    # sin_wave_2 = np.sin(n*(2*np.pi)*(f_targ_2/fs))
    sin_wave = audioSample(sin_wave_1, type="t", Fs=fs)

    pad = 10000
    # sin_wave.hanning()
    sin_wave.zeroPadStart(pad)
    sin_wave.zeroPadEnd(pad)

    sin_wave.plot(both=True)
    
    for theta in range(0,20,15):
        sin_shifted = mic.apply_xy(sin_wave, theta)

        sin_shifted.plot(both=True)


def simulate_polar_1mic():

    filename = "/home/terrasa/UROP/polar-measurement/data/19_Jan15/spv1840.pkl" 
    pd = polarData.fromPkl(filename)


    pd.plotAngle(0, both=True)




    # position in mm
    mic = Microphone(pd, [-500,2000])

    fs = 44.1e3
    length = 100000
    n = np.arange(10000)
    f_options = np.int32(np.logspace(2,4, 6))*2*(fs/length)

    plt.figure(1)

    for f_test in f_options:

        sin_wave = np.sin(n*(2*np.pi)*(f_test/fs))
        sin_wave = audioSample(sin_wave, type="t", Fs=fs)
        # sin_wave.hanning()


        thetas = np.array(list(range(0, 361, 1)))
        mags = []

        for theta in thetas:

            print(f_test, theta)

            result = mic.apply(sin_wave, theta)

            # sin_wave.plot(both=True) 
            # result.plot(both=True) 

            # get magnitude of the closest frequency of the result
            result.toDb()
            mags.append(result.getFreq([f_test])[0].real)

        plt.polar(thetas*np.pi/180, mags)


    plt.title("RESULT")
    plt.legend(np.int32(f_options), loc = "upper left")
    pd.plotFreqs(f_options, fig=2)



def simulate_polar_array():

    filename = "/home/terrasa/UROP/polar-measurement/data/19_Jan15/spv1840.pkl" 
    pd = polarData.fromPkl(filename)


    fs = 44.1e3
    length = 100000
    n = np.arange(10000)
    f_options = np.int32(np.logspace(2,4, 4))*2*(fs/length)

    # pd.plotAngle(90)

    f_1 = f_options[0]
    f_2 = f_options[2]
    c = 343e3
    d_1 = c/(2*f_1)
    d_2 = c/(2*f_2)

    # position in mm
    mic_1 = Microphone(pd, [0,0])
    mic_2 = Microphone(pd, [d_1, 0])
    mic_3 = Microphone(pd, [0, d_2])

    mic_array = MicrophoneArray([mic_1, mic_2])
    # mic_array = MicrophoneArray([mic_1, mic_2, mic_3])


    plt.figure(2)

    for f_test in f_options:

        sin_wave = np.sin(n*(2*np.pi)*(f_test/fs))
        sin_wave = 2/length*audioSample(sin_wave, type="t", Fs=fs)
        # sin_wave.hanning()


        thetas = np.array(list(range(0, 361, 2)))
        mags = []

        for theta in thetas:

            print(f_test, theta)

            result = mic_array.apply(sin_wave, theta)

            # sin_wave.plot(both=True) 
            # result.plot(both=True) 

            # get magnitude of the closest frequency of the result
            result.toDb()
            mags.append(result.getFreq([f_test])[0].real)

        plt.polar(thetas*np.pi/180, mags)

    plt.legend(f_options, loc = "upper left")
    mic_array.visualize()


def test_tf_prep():
    
    pd = polarData.fromPkl(filename)

    mic = Microphone(pd, (50,100))

    mic.polar[30].plot(both=True)
    
    mic.self_apply_xy()

    mic.polar[30].plot(both=True)

    angles, freqs, data = mic.tf_prep()

    print(angles)
    print(freqs)
    print(data)

    angles = tf.constant(angles)
    freqs = tf.constant(freqs)
    data = tf.constant(data)

    print(angles)
    print(freqs)
    print(data)






if __name__ == "__main__":
    # test_mic_apply()
    # test_xy()
    # simulate_polar_array()
    # simulate_polar_1mic()
    test_tf_prep()
