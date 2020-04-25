import numpy as np
from pythonAudioMeasurements.polarData import polarData
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

        original_type = signal.type

        signal.toTime()
        result = np.zeros(len(signal))

        for mic in self.microphones:
            this_result = mic.apply(signal, theta)
            this_result.toTime()
            result += this_result.data[:len(signal)] # accounts for 1-off from even-lengthsed signals

        signal.setType(original_type)

        result /= len(self.microphones)

        return audioSample(result, type=signal.type, Fs=signal.fs)

    def apply_xy(self, signal, theta):

        signal.toTime()
        result = np.zeros(len(signal))

        for mic in self.microphones:
            this_result = mic.apply_xy(signal, theta)
            this_result.toTime()
            result += this_result.data[:len(signal)] # accounts for 1-off from even-lengthsed signals

        return audioSample(result, type=signal.type, Fs=signal.fs)


    def visualize(self, fig=1):

        x = [mic.position[0] for mic in self.microphones]
        y = [mic.position[1] for mic in self.microphones]

        plt.figure(fig)
        plt.plot(x, y, "b*")

        plt.show()


    def tf_prep(self):

        freqs = self.microphones[0].polar.f()
        angles = self.microphones[0].polar.angles
        mic_responses = []

        for mic in self.microphones:

            angles_this_mic, freqs_this_mic, response = mic.tf_prep()

            assert all(np.abs(freqs_this_mic - freqs) < 1e-10), \
                "All microphones must have a response of the same length"

            assert all(np.abs(angles_this_mic - angles) < 1e-10), \
                "All microphones be sampled for the same angles"

            mic_responses.append(response)


        return angles, freqs, mic_responses

            
