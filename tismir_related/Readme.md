### Requirements
* Install the packages in [requirements.txt](requirements.txt)

### Usage
* Download and extract the contents of the repository
* Change directory to 4way-tabla-transcription/tismir_related
* To perform stroke transcription on a query audio file, run:  
```
python3 transcribe.py --input <path/to/test/audio> --output <path/to/folder/to/save/output>
```

The output transcription is a 2-column text file containing detected onset locations (in seconds) and the corresponding stroke category labels.

* Additionally, the threshold used for peak-picking (0.3 by default) and a gpu device ID (if gpu is available) can be set by passing the below arguments to the transcribe.py command:
```
--threshold <value> --gpu_id <ID>
```
