import os
import numpy as np
import torch
import utils
import argparse
from pathlib import PurePath

def classify_strokes(path_to_audio, predict_thresh, combine_thresh, saved_model_dir, device):
	categories = ['D', 'RT', 'RB']
	model_names = {'D': utils.onsetCNN_D(), 'RT': utils.onsetCNN_RT(), 'RB': utils.onsetCNN_RB()}
	folds = [0,1,2]
	seq_length = 15
	hop_dur = 10e-3

	#get log-mel-spectrogram of audio
	melgrams = utils.gen_melgrams(path_to_audio)

	#get frame-wise onset predictions
	n_frames = melgrams.shape[-1]-seq_length
	odf = dict(zip(categories, [np.zeros(n_frames)]*4))

	x = torch.tensor(melgrams[:,:,:]).double().to(device)
	x = x.unsqueeze(0)

	for cat in categories:
		y=0
		for fold in folds:
			saved_model_path = os.path.join(saved_model_dir, cat, 'saved_model_%d.pt'%fold)
			model = model_names[cat].double().to(device)
			model.load_state_dict(torch.load(saved_model_path, map_location=device))
			model.eval()

			y += model(x).squeeze().cpu().detach().numpy()
		odf[cat] = y/len(folds)

	#pick peaks in predicted activations
	odf_peaks = dict(zip(categories, []*4))
	for cat in categories:
		odf_peaks[cat] = utils.peakPicker(odf[cat], predict_thresh)
		odf_peaks[cat] = np.array(odf_peaks[cat],dtype=float)*hop_dur

	odf_peaks['D'] = utils.remove_damped_onsets(odf_peaks['D'], odf_peaks['RB'], odf_peaks['RT'], combine_thresh)
	odf_peaks['B'], odf_peaks['RB'], odf_peaks['RT'] = utils.return_combined_onsets(odf_peaks['RB'], odf_peaks['RT'], combine_thresh)

	onsets = np.concatenate([odf_peaks[cat] for cat in odf_peaks])
	labels = np.concatenate([[cat]*len(odf_peaks[cat]) for cat in odf_peaks])

	sorted_order = onsets.argsort()
	onsets = onsets[sorted_order]
	labels = labels[sorted_order]

	return onsets, labels, odf_peaks
	
if __name__=='__main__':
	parser = argparse.ArgumentParser(description="tabla stroke classifier")
	parser.add_argument("--input", type=str, default="", help="path to input audio")
	parser.add_argument("--output", type=str, default="../outputs/", help="folder to save output transcription")
	parser.add_argument("--threshold", type=float, default=0.3, help="threshold for peak-picking")
	parser.add_argument("--comb_threshold", type=float, default=40e-3, help="temporal threshold for combining simultaneous onsets of D and RT/RB to RT/RB or to combine simultaneous onsets of RT and RB to B")
	parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")

	args, _ = parser.parse_known_args()

	fs = 16000
	device = torch.device("cuda:%d"%args.gpu_id if torch.cuda.is_available() else "cpu")
	saved_model_dir = './saved_models'

	# Get 4-way tabla stroke classification outputs
	onsets, labels, odf_peaks = classify_strokes(args.input, args.threshold, args.comb_threshold, saved_model_dir, device)
		
	# Write combined transcript with all stroke categories
	outdir = '%s/combined'%args.output
	output_file_name = PurePath(args.input).name.replace(PurePath(args.input).suffix,'.txt')
	if not os.path.exists(outdir): os.makedirs(outdir)
	with open(os.path.join(outdir, output_file_name),'w') as fout:
		for time,label in np.nditer([onsets,labels]):
			fout.write('%f \t %s \n'%(time, label))

	# Write onsets (timestamps only) of each stroke category separately
	for cat in odf_peaks:
		outdir = '%s/%s'%(args.output,cat)
		output_file_name = PurePath(args.input).name.replace(PurePath(args.input).suffix,'_%s.txt'%cat)		
		if not os.path.exists(outdir): os.makedirs(outdir)
		np.savetxt('%s/%s'%(outdir,output_file_name),odf_peaks[cat])
