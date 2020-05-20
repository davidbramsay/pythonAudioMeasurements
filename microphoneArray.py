"""

Contains functionality for each arrays of Microphone objects

"""
import numpy as np
from pythonAudioMeasurements.polarData import polarData
from pythonAudioMeasurements.audioSample import audioSample
import matplotlib.pyplot as plt
from scipy.signal import convolve



class MicrophoneArray:

    def __init__(self, microphones):
        """
        
        Simulates a microphone array
        
        
        ---------------------------------------------------------------------
        INPUTS
        ---------------------------------------------------------------------
        microphones		| (list) of Microphone objects
        ---------------------------------------------------------------------
        
        
        ---------------------------------------------------------------------
        OUTPUTS
        ---------------------------------------------------------------------
        				| () 
        ---------------------------------------------------------------------
        
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
        """

        Simulate running the given signal at an incedent angle of theta
        
        
        ---------------------------------------------------------------------
        INPUTS
        ---------------------------------------------------------------------
        signal			| (audioSample) input signal
        ---------------------------------------------------------------------
        theta			| (float, int) incedent angle
        ---------------------------------------------------------------------
        
        
        ---------------------------------------------------------------------
        OUTPUTS
        ---------------------------------------------------------------------
        (audioSample) result of simulating the signal 
        ---------------------------------------------------------------------

        """

        original_type = signal.type
        signal.toTime()
        result = np.zeros(len(signal))

        for mic in self.microphones:

            # apply this mic's transfer function to the signal
            this_result = mic.apply_xy(signal, theta)
            this_result.toTime()

            # add the result of each microphonw together
            result += this_result.data[:len(signal)] # accounts for 1-off from even-lengthsed signals

        # return input signal to original type
        signal.setType(original_type)

        return audioSample(result, type=signal.type, Fs=signal.fs)


    def visualize(self, fig=1):
        """

        Create a figure with marks at the locations of each microphone
        
        
        ---------------------------------------------------------------------
        INPUTS
        ---------------------------------------------------------------------
        fig				| (int) what fig to put the visualization on
        ---------------------------------------------------------------------
        
        """

        x = [mic.position[0] for mic in self.microphones]
        y = [mic.position[1] for mic in self.microphones]

        plt.figure(fig)
        plt.plot(x, y, "b*")

        plt.show()


    def tf_prep(self):
        """

        Convert this MicrophoneArray into the formats necessary for input 
        into a tensorflow model. That is each microphone is a AxF matrix
        where F is the number of frequencies and A is the number of angles 
        in the microphone. The angles, frequiences, and a list of 2D 
        microphone responses are returned  
        
        
        ---------------------------------------------------------------------
        OUTPUTS
        ---------------------------------------------------------------------
        angles			| (numpy.array) of the angles at which the 
                        | microphones where measured
        ---------------------------------------------------------------------
        freqs			| (numpy.array) of the frequencies at which the 
                        | microphon transfer function is calculated
        ---------------------------------------------------------------------
        mic_responses	| (list) of AxF arrays containings the transfer 
                        | function of each microphone at each angle
        ---------------------------------------------------------------------

        """

        # use the first microphone to get the frequencies and angles collected
        freqs = self.microphones[0].polar.f()
        angles = self.microphones[0].polar.angles
        mic_responses = []

        for mic in self.microphones:

            angles_this_mic, freqs_this_mic, response = mic.tf_prep()

            assert all(np.abs(freqs_this_mic - freqs) < 1e-10), \
                "All microphones must have a response of the same length"

            assert all(np.abs(angles_this_mic - angles) < 1e-10), \
                "All microphones must be sampled for the same angles"

            mic_responses.append(response)


        return angles, freqs, mic_responses

            
