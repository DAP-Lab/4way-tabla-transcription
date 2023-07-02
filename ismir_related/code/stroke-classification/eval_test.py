import sys
import glob
import torch
from torch.utils import data
import numpy as np
import os
import utils
import mir_eval
import h5py as h5
from tqdm import tqdm
import argparse

#analysis parameters
hop_dur = 10e-3
seq_dur = 150e-3
seq_length = int(np.floor(seq_dur/hop_dur))
n_channels = 3
n_mels = 80

#evaluation tolerance
tolerance = 50e-3 #+- tolerance/2 seconds
tolerance_frame = tolerance/hop_dur

parser = argparse.ArgumentParser(description="tabla stroke classifier evaluation")
parser.add_argument("--stroke", type=str, default="", help="stroke category to evaluate for: d, rt, rb or b")
parser.add_argument("--expt_no", type=int, default=0, help="expt number in name of folder used to save trained model")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
parser.add_argument("--datapath", type=str, default="../dataset/test/melgrams", help="path to melgrams and labels saved by make_data.py")
parser.add_argument("--GTpath", type=str, default="../dataset/test/onsets", help="path to ground truth onsets")
parser.add_argument("--saved_model_path", type=str, default="../saved_models", help="path to saved trained model")
parser.add_argument("--pp_thresh", type=float, default=0.3, help="peak-picking threshold based on best CV score")

args, _ = parser.parse_known_args()

#Use gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:%d"%args.gpu_id if use_cuda else "cpu")

#model
if args.stroke=='d':
	from utils import onsetCNN_D as onsetCNN
elif args.stroke=='rt':
	from utils import onsetCNN_RT as onsetCNN
else:   
	from utils import onsetCNN

songlist = np.loadtxt('./cv_folds/test.fold',dtype=str)

#loop over test songs
n_folds=3
scores=[]
for song in tqdm(songlist):

	#predict frame-wise labels for song
	with h5.File(os.path.join(args.datapath, song+'.hdf5'), 'r') as hf:
		x=torch.tensor(hf['data'][:]).double().to(device)
	x=x.unsqueeze(0)
	n_frames = x.shape[-1]-(seq_length-1)

	odf=np.array([])
	y=np.zeros(n_frames)
	for fold in range(n_folds):
		path_to_saved_model = os.path.join(args.saved_model_path, args.stroke, 'expt_%d'%args.expt_no, 'saved_model_%d.pt'%fold)
		model = onsetCNN().double().to(device)
		model.load_state_dict(torch.load(path_to_saved_model, map_location=device))
		model.eval()

		y += model(x).squeeze().cpu().detach().numpy()

	odf = y/n_folds

	#load ground truth onset time-stamps
	gt = np.loadtxt(os.path.join(args.GTpath,args.stroke,song+'.onsets'))

	#pick peaks on predicted odf
	odf_labels = utils.peakPicker(odf,args.pp_thresh)
	odf_labels = np.array(odf_labels,dtype=float)*hop_dur

	#evaluate
	if odf_labels.size==1: odf_labels=np.array([odf_labels])
	if gt.size==1: gt=np.ravel(gt)
	scores.append(mir_eval.onset.f_measure(gt,odf_labels)[0])

#compute avg. f-score over test set
scores=np.mean(scores)
print('Test set f-score:\n')
print(scores)
