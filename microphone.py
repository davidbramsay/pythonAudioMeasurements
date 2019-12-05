import numpy as np
from polarData import polarData
from pythonAudioMeasurements.audioSample import audioSample
import matplotlib.pyplot as plt
from scipy.signal import convolve




class Microphone:

    def __init__(self, polar, position=[0,0], c=343e3):
        """
        position should be a vector x,y of the relative positoin
        from the specified origen
        position, c must have the same distance units, defaults assume mm
        """
        
        self.polar = polar
        self.position = np.array(position)
        self.c = c

    def normal_origin_dist(self, theta):
        """
        theta is in degrees

        if a plane wave is traveling in at an angle theta, 
        calculates the distance from the origien that the microphone is 
        normal to sound wave

        this is the "addition distance" covered by a plane wave between
        when the mirophone experiences the sound and when it would be 
        experienced at the origen
        """

        # conversion to radians
        theta *= (np.pi/180)

        # unit vector antiparellel to wave direction (from origin to plane)
        plane_direction = np.array([np.cos(theta), np.sin(theta)])

        return np.dot(plane_direction, self.position)


    def apply(self, signal, theta):

        mic = self.apply_microphone(signal, theta)
        return self.apply_xy(mic, theta)


    def apply_xy(self, signal, theta):
        """
        theta is given in degrees 

        applies the resulting phase shift to a signal based on the 
        x and y position of the micriophone this effectively 
        collapses the microphone to the origen

        signal should be an audioSample object

        leverages that in the freq domain e^(j*t_0*w)*X(w) shifts the
        corresponding time domain signal by t_0


        returns an object of type audioSample
        """

        signal.toFreq()
        freqs = signal.f()

        print(self.normal_origin_dist(theta))

        # time-domain shift
        delta_t =  self.normal_origin_dist(theta)/self.c

        phase_shift = np.exp(-1j*np.pi*freqs*delta_t)

        return signal*phase_shift


    def apply_microphone(self, signal, theta, f_targ=None):
        """
        the input signal must have been taken at the same sample rate
        for this to work. 
        """

        # get the frequency response of the microphone at the given theta
        mic_response = self.polar.getData(theta)
        mic_response.removeDCOffset()

        if f_targ:
            for i, f in enumerate(mic_response.f()):
                if f >= f_targ: 
                    mic_response.toFreq()
                    print(f, abs(mic_response[i]))
                    break

        # must have signal of same fs as mic response
        assert mic_response.fs == signal.fs, "your input signal must have the same " + \
        "sampling frequency. the microphone has fs: %d, and your signal as %d"%(mic_response.fs, signal.fs)

        
        signal.toTime()
        mic_response.toTime()

        result = convolve(signal, mic_response, "same")

        print(max(signal))
        # print(np.average(signal))
        # print(sum(signal))
        print(max(result))
        # print(np.average(result))
        # print(sum(result))

    
        return audioSample(result, Fs=signal.fs) 



if __name__ == "__main__":
    pass