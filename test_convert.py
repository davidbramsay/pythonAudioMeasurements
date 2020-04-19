import pickle
from pythonAudioMeasurements.audioSample import audioSample
from pythonAudioMeasurements.polarData import polarData

filename = "/home/terrasa/UROP/polar-measurement/data/19_Jan15_fixedpkls/spv1840.pkl"
filename_2 = "/home/terrasa/UROP/polar-measurement/data/19_Jan15/spv1840.pkl"


with open(filename, "rb") as this_file:
    # should be a dictionary
    raw_load = pickle.load(this_file)

test_samp = raw_load["measurements"][5]
print(test_samp)
as_asamp = audioSample(test_samp[0], test_samp[1], test_samp[2])
print(as_asamp)


as_asamp.plot(both=True)


pd = polarData.fromPkl(filename)
pd_2 = polarData.fromPkl(filename_2, pickleAsAudioSample=True)

pd.plotFreqs([400, 1000])
pd_2.plotFreqs([400, 1000])