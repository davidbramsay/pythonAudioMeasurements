import numpy as np
from polarData import polarData
from pythonAudioMeasurements.audioSample import audioSample
import matplotlib.pyplot as plt
from scipy.signal import convolve



class MicrophoneArray:

    def __init__(self, microphones):
        """
        microphone is a list of Microphone objects
        """

        self.microphones = microphones



    def apply(self, signal, theta):

        result = np.zeros(len(signal))
        print(len(signal))

        for mic in self.microphones:
            this_result = mic.apply(signal, theta)
            this_result.toTime()
            # print(len(this_result))
            result += this_result.data[:len(signal)] # accounts for 1-off from even-lengthsed signals

        return audioSample(result, type=signal.type, Fs=signal.fs)


