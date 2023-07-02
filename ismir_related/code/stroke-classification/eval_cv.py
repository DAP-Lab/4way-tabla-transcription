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
parser.add_argument("--expt_no", type=int, default=0, help="expt number in name of folder used to save trained model")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
parser.add_argument("--stroke", type=str, default="", help="stroke category to evaluate for: d, rt, rb or b")
parser.add_argument("--fold", type=int, default=0, help="cross-validation fold to evaluate on")
parser.add_argument("--datapath", type=str, default="../dataset/train/melgrams", help="path to melgrams and labels saved by make_data.py")
parser.add_argument("--GTpath", type=str, default="../dataset/train/onsets", help="path to ground truth onsets")
parser.add_argument("--saved_model_path", type=str, default="../saved_models", help="path to saved trained model")

args, _ = parser.parse_known_args()

#model
if args.stroke=='d':
	from utils import onsetCNN_D as onsetCNN
elif args.stroke=='rt':
	from utils import onsetCNN_RT as onsetCNN
else:   
	from utils import onsetCNN

#Use gpu if present
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:%d"%args.gpu_id if use_cuda else "cpu")

path_to_saved_model = os.path.join(args.saved_model_path, args.stroke, 'expt_%d'%args.expt_no, 'saved_model_%d.pt'%args.fold)
songlist = np.loadtxt('./cv_folds/3fold_cv_%d.fold'%args.fold,dtype=str)

#load model
model = onsetCNN().double().to(device)
model.load_state_dict(torch.load(path_to_saved_model))
model.eval()

#loop over val set songs
scores=np.array([])
for song in tqdm(songlist):
	#load input melgram for full song
	with h5.File(os.path.join(args.datapath, song+'.hdf5'), 'r') as hf:
		x=torch.tensor(hf['data'][:]).double().to(device)
		n_frames = hf['data'].shape[-1]-seq_length
	x=x.unsqueeze(0)

	#model forward pass
	odf = model(x).squeeze().cpu().detach().numpy()

	#load ground truth onset time-stamps
	gt = np.loadtxt(os.path.join(args.GTpath,args.stroke,song+'.onsets'))

	#evaluate odf - loop over different peak-picking thresholds to optimize F-score
	scores_thresh=np.array([])
	for predict_thresh in [0.3, 0.4, 0.5]:
		#pick peaks
		odf_labels = utils.peakPicker(odf,predict_thresh)
		odf_labels = np.array(odf_labels,dtype=float)*hop_dur

		#evaluate
		if odf_labels.size==1: odf_labels=np.array([odf_labels])
		if gt.size==1: gt=np.ravel(gt)
		scores_thresh = np.append(scores_thresh, mir_eval.onset.f_measure(gt,odf_labels)[0])

	#accumulate for every song
	if len(scores)==0: scores=np.atleast_2d(np.array(scores_thresh))
	else: scores=np.vstack((scores,np.atleast_2d(np.array(scores_thresh))))

#compute avg. f-score over val set
scores=np.mean(scores,0)
print('f-scores at different peak-picking thresholds:\n')
print(scores)
