import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import torch
from torch.utils import data as th_data
from utils import Dataset
import argparse
import utils

def train(model, optimizer, criterion, training_generator):
	model.train()
	n_batch=0
	loss_epoch=0
	for local_batch, local_labels, local_weights in training_generator:
		n_batch+=1

		#transfer to GPU
		local_batch, local_labels, local_weights = local_batch.double().to(device), local_labels.double().to(device), local_weights.double().to(device)

		#model forward pass
		optimizer.zero_grad()
		outs = model(local_batch).squeeze()
		outs = outs.double()

		#compute loss
		loss = criterion(outs, local_labels)
		loss = loss.double()
		loss = torch.dot(loss,local_weights)
		loss /= local_batch.size()[0]
		loss_epoch+=loss.item()

		#update weights
		loss.backward()
		optimizer.step()
	return model, loss_epoch/n_batch

def validate(model, criterion, validation_generator):
	model.eval()
	n_batch=0
	loss_epoch=0
	with torch.set_grad_enabled(False):
		for local_batch, local_labels, local_weights in validation_generator:
			n_batch+=1

			#transfer to GPU
			local_batch, local_labels = local_batch.double().to(device), local_labels.double().to(device)

			#model forward pass
			outs = model(local_batch).squeeze()
			outs=outs.double()

			#compute loss
			loss = criterion(outs, local_labels).mean()
			loss_epoch+=loss.item()
	return model, loss_epoch/n_batch

def plot_losses(train_loss,val_loss,save_filepath):
	plt.plot(train_loss,label='train')
	plt.plot(val_loss,label='val')
	plt.legend()
	plt.savefig(save_filepath)
	plt.clf()
	return

if __name__ == "__main__" :
	parser = argparse.ArgumentParser(description="tabla stroke classifier training")
	parser.add_argument("--expt_no", type=int, default=0, help="expt number to name folder used to save trained model")
	parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
	parser.add_argument("--stroke", type=str, default="", help="stroke category to train for: d, rt, rb or b")
	parser.add_argument("--fold", type=int, default=0, help="cross-validation fold to hold out: 0, 1 or 2")
	parser.add_argument("--datapath", type=str, default="", help="path to melgrams and labels saved by make_data.py")
	parser.add_argument("--aug_method", type=str, default=None, help="augmentation method: 'orig', 'ps', 'ts', 'ss-all', 'sr-all'")
	parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
	parser.add_argument("--num_workers", type=int, default=4, help="number of parallel data loading workers")
	parser.add_argument("--max_epochs", type=int, default=50, help="maximum number of training epochs")

	args, _ = parser.parse_known_args()

	#paths to save training outputs
	model_save_dir = '../saved_models/%s/expt_%d'%(args.stroke,args.expt_no)
	if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
	plot_save_dir = '../plots/%s/expt_%d'%(args.stroke,args.expt_no)
	if not os.path.exists(plot_save_dir): os.makedirs(plot_save_dir)

	#use GPU if present
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:%d"%args.gpu_id if use_cuda else "cpu")

	#model
	if args.stroke=='d':
		from utils import onsetCNN_D as onsetCNN
	elif args.stroke=='rt':
		from utils import onsetCNN_RT as onsetCNN
	else:
		from utils import onsetCNN

	model=onsetCNN().double().to(device)
	criterion=torch.nn.BCELoss(reduction='none')
	optimizer=torch.optim.Adam(model.parameters(), lr=1e-4)

	#make train-val splits
	print('\n---\nMaking train-val splits\n---\n')
	train_val_data_filepaths = {'train':os.path.join(args.datapath, 'labels_weights_train_%s.hdf5'%args.aug_method), 'validation':os.path.join(args.datapath, 'labels_weights_val_%s.hdf5'%args.aug_method)} 	# these are temp files that will be created and overwritten during every training run; they contain the frame-wise onset labels for a given stroke and weights to be applied during loss computation 

	#get list of audios in each CV fold
	split_dir = './cv_folds'
	folds = {'val': args.fold, 'train': np.delete([0,1,2], args.fold)}
	splits = dict(zip([0, 1, 2], [np.loadtxt(os.path.join(split_dir, '3fold_cv_%d.fold'%fold), dtype=str) for fold in range(3)]))

	#create training and validation splits of data and save them to disk as temporary files
	labels_weights_orig_filepath = os.path.join(args.datapath, 'labels_weights_orig_%s.hdf5'%args.stroke)
	labels_weights_aug_filepath = os.path.join(args.datapath, 'labels_weights_%s_%s.hdf5'%(args.aug_method,args.stroke))
	utils.make_train_val_split(folds, labels_weights_orig_filepath, labels_weights_aug_filepath, train_val_data_filepaths)

	#load all melgram-label pairs as a dict to memory for faster training (ensure sufficient RAM size apriori)
	songlist_orig = np.loadtxt('./songlists/songlist_orig.txt', dtype=str)
	songlist_aug = np.loadtxt('./songlists/songlist_%s.txt'%args.aug_method, dtype=str)
	mel_data = utils.load_mel_data(args.datapath, folds, splits, songlist_orig, songlist_aug)

	#data loaders
	params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.num_workers}
	training_set = Dataset(train_val_data_filepaths['train'], seq_length=15, n_channels=3, mel_data=mel_data)
	training_generator = th_data.DataLoader(training_set, **params)

	validation_set = Dataset(train_val_data_filepaths['validation'], seq_length=15, n_channels=3, mel_data=mel_data)
	validation_generator = th_data.DataLoader(validation_set, **params)

	#early stop parameters
	early_stop_count = 0
	early_stop_patience = 10

	#train-val loop
	print('\n---\nStarting training\n---\n')
	train_loss_epoch=[]
	val_loss_epoch=[]
	for epoch in range(args.max_epochs):
		##training
		model, loss_epoch = train(model, optimizer, criterion, training_generator)
		train_loss_epoch.append(loss_epoch)

		##validation
		model, loss_epoch = validate(model, criterion, validation_generator)
		val_loss_epoch.append(loss_epoch)

		#print loss in current epoch
		print('Epoch no: %d/%d\tTrain loss: %f\tVal loss: %f'%(epoch, args.max_epochs, train_loss_epoch[-1], val_loss_epoch[-1]))

		#save model if val loss is minimum so far
		if(val_loss_epoch[-1] == min(val_loss_epoch)):
			torch.save(model.state_dict(), os.path.join(model_save_dir, 'saved_model_%d.pt'%args.fold))

			#reset early stop count to 0 since new minimum loss reached
			early_stop_count = 0

		else:
			early_stop_count += 1

		if(early_stop_count == early_stop_patience):
			print('Early stopping at %d'%epoch)
			break

	#plot losses vs epoch
	plot_save_filepath = os.path.join(plot_save_dir, 'loss_curves_%d.png'%args.fold)
	plot_losses(train_loss_epoch,val_loss_epoch,plot_save_filepath)
