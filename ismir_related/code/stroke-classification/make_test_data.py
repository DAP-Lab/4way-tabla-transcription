import numpy as np
import os
import librosa
import torch
import sys
import h5py as h5
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="test set input data creator")
parser.add_argument("--datapath", type=str, default="../dataset/test", help="path to base directory where test set audios & onsets are saved")

args, _ = parser.parse_known_args()

#data dirs
audio_dir = os.path.join(args.datapath,'audios')
save_dir = os.path.join(args.datapath,'melgrams')
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

#data stats for normalization
stats=np.load('./means_stds.npy')
means=stats[0]
stds=stats[1]

#context parameters
contextlen = 7 #+- frames
duration = 2*contextlen+1

#analysis parameters
fs=16000
hopDur = 10e-3
hopSize = int(np.ceil(hopDur*fs))
winDur_list = [23.2e-3, 46.4e-3, 92.8e-3]
winSize_list = [int(np.ceil(winDur*fs)) for winDur in winDur_list]
nFFT_list = [2**(int(np.ceil(np.log2(winSize)))) for winSize in winSize_list]
fMin=27.5
fMax=8000
nMels=80

#cv-splits
split_dir = './cv_folds'
songlist = np.loadtxt(os.path.join(split_dir, 'test.fold'), dtype=str)

#main
for song in tqdm(songlist):
	suffix='.wav' if '.wav' not in song else ''
	x,fs = librosa.load(os.path.join(audio_dir, song + suffix), sr=fs)

	#get mel spectrograms
	melgram1=librosa.feature.melspectrogram(x,sr=fs,n_fft=nFFT_list[0], win_length=winSize_list[0], hop_length=hopSize, n_mels=nMels, fmin=fMin, fmax=fMax)
	melgram2=librosa.feature.melspectrogram(x,sr=fs,n_fft=nFFT_list[1], win_length=winSize_list[1], hop_length=hopSize, n_mels=nMels, fmin=fMin, fmax=fMax)
	melgram3=librosa.feature.melspectrogram(x,sr=fs,n_fft=nFFT_list[2], win_length=winSize_list[2], hop_length=hopSize, n_mels=nMels, fmin=fMin, fmax=fMax)

	melgrams = np.array([melgram1, melgram2, melgram3])

	#log scaling
	melgrams=10*np.log10(1e-10+melgrams)

	#normalize
	melgrams = (melgrams - np.repeat(np.atleast_3d(means), melgrams.shape[2], axis=-1))/np.repeat(np.atleast_3d(stds), melgrams.shape[2], axis=-1)

	#zero pad ends
	melgrams = np.concatenate((np.zeros([melgrams.shape[0], melgrams.shape[1], contextlen]), melgrams, np.zeros([melgrams.shape[0], melgrams.shape[1], contextlen])), -1)

	#save melgram as hdf5 file
	save_path = os.path.join(save_dir, song + '.hdf5')
	with h5.File(save_path, 'w') as hf_melgram:
		hf_melgram.create_dataset('data', data=melgrams)
