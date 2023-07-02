How to use:

``` 
python3 augment_data_<method>.py --input <path/to/audio/file> --output <path/to/audio/file> --params <list of parameter values> --n_jobs <# CPU cores to use> 
```

Where <method> is one of: 'ps', 'ts', 'ar', 'sf', 'sr'. Refer to Section 3.2 of the paper for details on each method.

The list of parameter values must be separated by a space for the methods 'ps', 'ts', & 'ar'.
For 'sf' & 'sr', provide 3-element comma-separated tuples that are separated by a space.

Each parameter value (or set) results in one modified version of the original audio.

e.g.:

  ```python3 augment_data_ps.py --input ~/test_audio.wav --output ~/ --params 0.3 0.5 1.2 1.5 --n_jobs 4```
  
  ```python3 augment_data_sf.py --input ~/test_audio.wav --output ~/ --params 0.3,1.3,2.5 0.5,1.2,1.5 --n_jobs 2```

Further, the 'sr' method requires the path to saved templates to be provided. By default, it takes the sample templates file provided with this code. It also requires the NMF toolbox to be downloaded (from https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox/#Python) and the contents unpacked to the folder 'NMFToolbox' in the present directory.
