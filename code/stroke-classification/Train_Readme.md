### Readme for the training scripts

Instructions:
1. Run ```make_train_data.py``` and ```make_test_data.py``` first to create the input-output data for model training and testing.
  * Both scripts require the path to the tabla dataset of audios and onsets.
  * The ```make_train_data.py``` script additionally allows to specify an augmentation method to use for training data. Set the augmentation method to 'orig' for no augmentation.
  * The ```make_test_data.py``` script uses the list of tracks specified in the ```test.fold``` file saved in the ```cv_folds``` directory.
  * The ```make_train_data.py``` script uses different songlists (text files with a list of audio filenames without the full path) depending on augmentation method. Precomputed songlists are provided in the ```songlists``` directory and can be used if augmentation is being performed using the same modification parameter values as in the paper. If different values are used, then new songlists need to be created containing a list of all the audio filenames to be used (original + augmented).
  * The data created by these scripts are saved in a separate folder called ```melgrams``` within the train and test sub-directories of the base dataset folder (see expected folder structure below)  
  
2. To use augmented data for training, first generate augmented versions of the train set audios using the data augmentation scripts from this repository and then call the ```make_train_data.py``` script. Augmented audios are expected to be saved at the same path as the original audios, but in a different folder called ```audios_augmented```.
  
3. Next, run ```train.py```. The trained model and a plot of the loss curves will be saved in new folders created in this directory.
  
4. Train on all three CV folds.
  
5. Then, run the ```eval_cv.py``` and ```eval_test.py``` scripts to get CV and test set f-scores.  

Each script can be called with -h as a command-line argument to see the list of required arguments (e.g., ```python make_train_data.py -h```)

====  
Expected folder structure for the code:  
.  
├── 4way-tabla-transcription  
│   ├── code  
│   │   ├── stroke-classification  
│   │   │   ├── \*.py (scripts)  
│   │   │   ├── Train_readme.md  
│   │   │   ├── cv_folds/  
│   │   │   ├── songlists/  
│   │   ├── data-augmentation  
...  
  
====  
Expected folder structure for tabla dataset:  
.  
├── \<base-folder\>  
│   ├── train  
│   │   ├── audios  
│   │   ├── onsets  
│   │   ├── melgrams (is created by make_train_data.py)  
│   ├── test  
│   │   ├── audios  
│   │   ├── onsets  
│   │   ├── melgrams(is created by make_test_data.py)  
