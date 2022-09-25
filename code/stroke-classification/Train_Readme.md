Readme for the tabla 4way stroke classifier training scripts
(Rohit M. A., Sep 25 2022)
============================================================

Instructions:
  1. Run make_train_data.py and make_test_data.py first to create the input-output data for model training and testing.
    * Both scripts require the path to the tabla dataset of audios and onsets. The make_train_data.py script additionally allows to specify which augmentation method to use for training data. Set the augmentation method to 'orig' for no augmentation.
    * The make_test_data.py script uses the list of tracks specified in the test.fold file saved in the cv_folds directory.
    * The make_train_data.py script uses different songlists (text files with a list of audio filenames without the full path) depending on augmentation method. These songlists are provided in the songlists directory for convenience. These can be used if augmentation is being performed using the same modification parameter values as in the paper. If different values are used, then new songlists need to be created containing a list of all the audio filenames to be used (original + augmented).
    * The data created by these scripts are saved in a separate folder called 'melgrams' within the train and test sub-directories of the base dataset folder 
  2. To use augmented data for training, first generate augmented versions of the train set audios using the data augmentation scripts from the repository and then call the make_train_data.py script. Augmented audios are expected to be saved at the same path as the original audios, but in a different folder called 'audios_augmented'.
  3. Next, run train.py. The trained model and a plot of the loss curves will be saved in new folders created one level above this code directory.
  4. Train on all three CV folds.
  5. Then, run the eval_cv.py and eval_test.py scripts to get CV and test set f-scores.

====  
Expected folder structure within this code directory:
|-- 4way-tabla-transcription
    |-- code
        |-- stroke-classification
            |-- train.py, utils.py, etc. (scripts)
            |-- cv_folds
            |-- songlists
            |-- Train_readme.md
        |-- data-augmentation
        ...

====
Expected folder structure for tabla dataset:
|-- <base-folder>
    |-- train
        |-- audios
        |-- onsets
        |-- melgrams (is created by make_train_data.py)
    |-- test
        |-- audios
        |-- onsets
        |-- melgrams (is created by make_test_data.py)
