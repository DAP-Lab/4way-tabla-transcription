import numpy as np
import os
import librosa
import torch
import sys
import h5py as h5
from tqdm import tqdm
import argparse

def make_labels(length, onsets, hopDur, smear=True):
	labels=np.zeros(length)
	weights=np.ones(length)
	idxs=np.array(np.round(onsets/hopDur),dtype=int)
	labels[idxs]=1

	if smear:
		#target smearing
		labels[idxs-1]=1
		labels[idxs+1]=1
		weights[idxs-1]=0.25
		weights[idxs+1]=0.25
	return labels, weights

#cmd-line argument
parser = argparse.ArgumentParser(description="train set input-output data creator")
parser.add_argument("--aug_method", type=str, default='orig', help="augmentation method: 'orig', 'ps', 'ts', 'ss-all', 'sr-all'")
parser.add_argument("--datapath", type=str, default="../dataset/train", help="path to base directory where audios & onsets are saved")

args, _ = parser.parse_known_args()

#data dirs
if args.aug_method=='orig':
	audio_dir = os.path.join(args.datapath,'audios')
else:
	audio_dir = os.path.join(args.datapath,'audios_augmented')

onset_dir = os.path.join(args.datapath,'onsets')
save_dir = os.path.join(args.datapath,'melgrams')
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

#data stats for normalization
stats=np.load('../means_stds.npy')
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
split_dir = '../cv_folds'
splits=dict(zip(['fold0', 'fold1', 'fold2'], [np.loadtxt(os.path.join(split_dir, '3fold_cv_%d.fold'%fold), dtype=str) for fold in range(3)]))

#main
songlist=np.loadtxt('../songlists/songlist_%s.txt'%args.aug_method,dtype=str)

stroke_categories=['d','rt','rb','b']

for cat in stroke_categories:
	labels_savepath = os.path.join(save_dir, 'labels_weights_%s_%s.hdf5'%(args.aug_method, cat))
	if os.path.exists(labels_savepath):
		os.system('rm %s'%labels_savepath)

for song in tqdm(songlist):
	#get original song name (leaving out augmentation part of filename)
	if ('_ar_' in song) | ('_ss-bass_' in song) | ('_ss-treble_' in song) | ('_ss-perc_' in song) | ('_ts_' in song) | ('_ps_' in song) | ('_sr-bass_' in song) | ('_sr-treble_' in song) | ('_sr-perc_' in song):
		song_orig = '_'.join(song.split('_')[:-2])
	elif ('_ss-all_' in song) | ('_sr-all_' in song):
		song_orig = '_'.join(song.split('_')[:-4])
	else:
		song_orig = song

	#determine fold
	if song_orig in splits['fold0']: fold='fold0'
	elif song_orig in splits['fold1']: fold='fold1'
	else: fold='fold2'

	labels=dict(zip(stroke_categories, [[]]*len(stroke_categories)))
	weights=dict(zip(stroke_categories, [[]]*len(stroke_categories)))

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

	#get frame-wise labels & weights from onsets and save separately
	for cat in stroke_categories:
		onsets=np.loadtxt(os.path.join(onset_dir,cat,song_orig+'.onsets'))

		#modify onset time stamp if audio is time-scaled (as in time-scaling augmentation) 
		if '_ts_' in song:
			ts_factor = float(song.split('_')[-1])
			onsets = onsets*ts_factor

		#get frame-wise labels, weights
		labels[cat], weights[cat] = make_labels(melgrams.shape[2]-2*contextlen, onsets, hopSize/fs)

		data = [[save_path.replace('.hdf5', ''), str(i), str(labels[cat][i]), str(weights[cat][i])] for i in range(melgrams.shape[2]-2*contextlen)]

		#write to separate file
		labels_savepath = os.path.join(save_dir, 'labels_weights_%s_%s.hdf5'%(args.aug_method,cat))
		with h5.File(labels_savepath, 'a') as hf_data:
			if '%s/data'%fold not in hf_data:
				hf_data.create_dataset('%s/data'%fold, data=data, maxshape=(None,None))
			else:
				hf_data['%s/data'%fold].resize((hf_data['%s/data'%fold].shape[0] + len(data)), axis = 0)
				hf_data['%s/data'%fold][-len(data):] = data
