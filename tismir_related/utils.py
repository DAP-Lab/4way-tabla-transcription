import os
import librosa
import torchaudio
import torch
import numpy as np
import mir_eval
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from collections import OrderedDict

#model
class onsetCNN_RB(nn.Module):
	def __init__(self):
		super(onsetCNN_RB, self).__init__()
		self.conv_seq_layer_names = ['conv1', 'bn1', 're1', 'pool1', 'conv2', 'bn2', 're2', 'pool2']
		self.conv_pool_seq = torch.nn.ModuleDict(OrderedDict(zip(self.conv_seq_layer_names, nn.Sequential(
		nn.Conv2d(3, 16, (3,7)),
		nn.BatchNorm2d(16),
		nn.ReLU(),
		nn.MaxPool2d((3,1)),
		nn.Conv2d(16, 32, 3),
		nn.BatchNorm2d(32),
		nn.ReLU(),
		nn.MaxPool2d((3,1))
		))))

		self.dense_seq_layer_names = ['do1', 'fc1', 'bn1', 're1', 'do2', 'fc2', 'bn2', 'sig']
		self.dense_seq = torch.nn.ModuleDict(OrderedDict(zip(self.dense_seq_layer_names, nn.Sequential(
		nn.Dropout(p=0.25),
		nn.Linear(32 * 7 * 8, 128),
		nn.BatchNorm1d(128),
		nn.ReLU(),
		nn.Dropout(p=0.25),
		nn.Linear(128,1),
		nn.BatchNorm1d(1),
		nn.Sigmoid()
		))))

	def forward(self, y):
		for layer in self.conv_pool_seq:
			y=self.conv_pool_seq[layer](y)

		if y.shape[-1]>7:
			y_shape = y.shape
			y = y.reshape(1,y_shape[1]*y_shape[2],y_shape[-1]).permute(0,2,1).unsqueeze(0)
			y = torch.nn.Unfold(kernel_size=(7,y_shape[1]*y_shape[2]))(y)
			y_shape_new = y.shape
			y = y.reshape(1,7,y_shape[1]*y_shape[2],y_shape_new[-1]).permute(0,2,1,3).reshape(1,7*y_shape[1]*y_shape[2],y_shape_new[-1]).squeeze(0)
			y = y.permute(1,0)

		else:
			y=y.view(-1,32*7*8)

		for layer in self.dense_seq:
			y=self.dense_seq[layer](y)

		return y

# model for damped category
class onsetCNN_D(nn.Module):
	def __init__(self):
		super(onsetCNN_D, self).__init__()
		self.conv_seq_layer_names = ['conv1', 'bn1', 're1', 'pool1', 'conv2', 'bn2', 're2', 'pool2']
		self.conv_pool_seq = torch.nn.ModuleDict(OrderedDict(zip(self.conv_seq_layer_names, nn.Sequential(
		nn.Conv2d(3, 16, (3,7)),
		nn.BatchNorm2d(16),
		nn.ReLU(),
		nn.MaxPool2d((3,1)),
		nn.Conv2d(16, 32, 3),
		nn.BatchNorm2d(32),
		nn.ReLU(),
		nn.MaxPool2d((3,1))
		))))

		self.dense_seq_layer_names = ['do1', 'fc1', 'bn1', 're1', 'do2', 'fc2', 'bn2', 'sig']
		self.dense_seq = torch.nn.ModuleDict(OrderedDict(zip(self.dense_seq_layer_names, nn.Sequential(
		nn.Dropout(p=0.25),
		nn.Linear(32 * 7 * 8, 256),
		nn.BatchNorm1d(256),
		nn.ReLU(),
		nn.Dropout(p=0.25),
		nn.Linear(256,1),
		nn.BatchNorm1d(1),
		nn.Sigmoid()
		))))

	def forward(self, y):
		for layer in self.conv_pool_seq:
			y=self.conv_pool_seq[layer](y)

		if y.shape[-1]>7:
			y_shape = y.shape
			y = y.reshape(1,y_shape[1]*y_shape[2],y_shape[-1]).permute(0,2,1).unsqueeze(0)
			y = torch.nn.Unfold(kernel_size=(7,y_shape[1]*y_shape[2]))(y)
			y_shape_new = y.shape
			y = y.reshape(1,7,y_shape[1]*y_shape[2],y_shape_new[-1]).permute(0,2,1,3).reshape(1,7*y_shape[1]*y_shape[2],y_shape_new[-1]).squeeze(0)
			y = y.permute(1,0)
		else:
			y=y.view(-1,32*7*8)

		for layer in self.dense_seq:
			y=self.dense_seq[layer](y)
		
		return y

# model for treble category
class onsetCNN_RT(nn.Module):
	def __init__(self):
		super(onsetCNN_RT, self).__init__()
		self.conv_seq_layer_names = ['conv1', 'bn1', 're1', 'pool1', 'conv2', 'bn2', 're2', 'pool2']
		self.conv_pool_seq = torch.nn.ModuleDict(OrderedDict(zip(self.conv_seq_layer_names, nn.Sequential(
		nn.Conv2d(3, 32, (3,7)),
		nn.BatchNorm2d(32),
		nn.ReLU(),
		nn.MaxPool2d((3,1)),
		nn.Conv2d(32, 64, 3),
		nn.BatchNorm2d(64),
		nn.ReLU(),
		nn.MaxPool2d((3,1))
		))))

		self.dense_seq_layer_names = ['do1', 'fc1', 'bn1', 're1', 'do2', 'fc2', 'bn2', 'sig']
		self.dense_seq = torch.nn.ModuleDict(OrderedDict(zip(self.dense_seq_layer_names, nn.Sequential(
		nn.Dropout(p=0.25),
		nn.Linear(64 * 7 * 8, 128),
		nn.BatchNorm1d(128),
		nn.ReLU(),
		nn.Dropout(p=0.25),
		nn.Linear(128,1),
		nn.BatchNorm1d(1),
		nn.Sigmoid()
		))))


	def forward(self, y):
		for layer in self.conv_pool_seq:
			y=self.conv_pool_seq[layer](y)

		if y.shape[-1]>7:
			y_shape = y.shape
			y = y.reshape(1,y_shape[1]*y_shape[2],y_shape[-1]).permute(0,2,1).unsqueeze(0)
			y = torch.nn.Unfold(kernel_size=(7,y_shape[1]*y_shape[2]))(y)
			y_shape_new = y.shape
			y = y.reshape(1,7,y_shape[1]*y_shape[2],y_shape_new[-1]).permute(0,2,1,3).reshape(1,7*y_shape[1]*y_shape[2],y_shape_new[-1]).squeeze(0)
			y = y.permute(1,0)
		else:
			y=y.view(-1,64*7*8) ### CHANGED_context

		for layer in self.dense_seq:
			y=self.dense_seq[layer](y)

		return y
#################################

#peak-picking function
def peakPicker(data, peakThresh):
	peaks=np.array([],dtype='int')
	for ind in range(1,len(data)-1):
		if ((data[ind+1] < data[ind] > data[ind-1]) & (data[ind]>peakThresh)):
			peaks=np.append(peaks,ind)
	return peaks

#merge onsets if too close - retain only stronger one
def merge_onsets(onsets,strengths,mergeDur):
	onsetLocs=np.where(onsets==1)[0]
	ind=1
	while ind<len(onsetLocs):
		if onsetLocs[ind]-onsetLocs[ind-1] < mergeDur:
			if strengths[onsetLocs[ind]]<strengths[onsetLocs[ind-1]]:
				onsets[onsetLocs[ind]]=0
				onsetLocs=np.delete(onsetLocs,ind)
			else:
				onsets[onsetLocs[ind-1]]=0
				onsetLocs=np.delete(onsetLocs,ind-1)
		else: ind+=1
	return onsets

#generate log-mel-spectrograms given path to audio
def gen_melgrams(path_to_audio):
	#analysis parameters
	fs=16000
	hopDur=10e-3
	hopSize = int(np.ceil(hopDur*fs))
	winDur_list = [23.2e-3, 46.4e-3, 92.8e-3]
	winSize_list = [int(np.ceil(winDur*fs)) for winDur in winDur_list]
	nFFT_list = [2**(int(np.ceil(np.log2(winSize)))) for winSize in winSize_list]
	fMin=27.5
	fMax=8000
	nMels=80

	#context parameters
	contextlen=7 #+- frames
	duration=2*contextlen+1

	#data stats for normalization
	stats=np.load('./means_stds.npy')
	means=stats[0]
	stds=stats[1]

	x,fs = librosa.load(path_to_audio, sr=fs)

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

	return melgrams
	
#############################################################################################################################################

# takes in onset labels of bass and treble, and returns labels of both, bass and treble
def return_combined_onsets(odf_labels_bass, odf_labels_treble, tolerance=40e-3):
	matched_onsets=[]
	if ( len(odf_labels_bass)>0 ) & ( len(odf_labels_treble)>0 ):
		matched_onsets = mir_eval.util.match_events(odf_labels_treble, odf_labels_bass, tolerance)
		#matched_onsets is a list of tuples (i,j) where odf_labels_treble[i] matches odf_labels_bass[j]

	if len(matched_onsets)==0:
			return np.asarray(matched_onsets), np.asarray(odf_labels_bass), np.asarray(odf_labels_treble)

	#remove merged onsets from rt and rb
	onset_seq_rt = [item for i,item in enumerate(odf_labels_treble) if i not in np.array(matched_onsets)[:,0]]
	onset_seq_rb = [item for i,item in enumerate(odf_labels_bass) if i not in np.array(matched_onsets)[:,1]]

	## Further processing to get resonant both onsets
	#first get onset times corresponding to matched onset indices
	matched_onset_times = np.asarray([(odf_labels_treble[item[0]], odf_labels_bass[item[1]]) for item in matched_onsets])
	#then choose earlier onset among rt and rb as onset of b
	matched_onset_times = np.min(matched_onset_times,1)

	return np.asarray(matched_onset_times), np.asarray(onset_seq_rb), np.asarray(onset_seq_rt)



# takes in onset labels of bass and treble and damped, and returns the damped onset labels with the coinciding damped labels removed
def remove_damped_onsets(odf_labels_damped, odf_labels_bass, odf_labels_treble, tolerance=40e-3):
	matched_onsets_bass=[]
	matched_onsets_treble=[]
	onset_seq_d = []

	if ( len(odf_labels_bass)>0 ) & ( len(odf_labels_damped)>0 ):
		matched_onsets_bass = mir_eval.util.match_events(odf_labels_damped, odf_labels_bass, tolerance)
		#matched_onsets_bass is a list of tuples (i,j) where odf_labels_damped[i] matches odf_labels_bass[j]

	if ( len(odf_labels_treble)>0 ) & ( len(odf_labels_damped)>0 ):
		matched_onsets_treble = mir_eval.util.match_events(odf_labels_damped, odf_labels_treble, tolerance)
		#matched_onsets_treble is a list of tuples (i,j) where odf_labels_damped[i] matches odf_labels_treble[j]

	if len(matched_onsets_bass)==0 and len(matched_onsets_treble)==0:
		return np.asarray(odf_labels_damped)

	elif len(matched_onsets_bass)!=0 and len(matched_onsets_treble)==0:
		onset_seq_d = [item for i,item in enumerate(odf_labels_damped) if i not in np.array(matched_onsets_bass)[:,0]]

	elif len(matched_onsets_bass)==0 and len(matched_onsets_treble)!=0:
		onset_seq_d = [item for i,item in enumerate(odf_labels_damped) if i not in np.array(matched_onsets_treble)[:,0]]
	
	elif len(matched_onsets_bass)!=0 and len(matched_onsets_treble)!=0:
		onset_seq_d = [item for i,item in enumerate(odf_labels_damped) if (i not in np.array(matched_onsets_bass)[:,0]) and (i not in np.array(matched_onsets_treble)[:,0])]
		
	return np.asarray(onset_seq_d)
