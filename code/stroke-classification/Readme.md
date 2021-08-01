How to use:

```
python3 transcribe.py --input <path/to/test/audio> --output <path/to/folder/to/save/output>
```

The output transcription is a 2-column text file containing detected onset locations (in seconds) and the corresponding stroke category labels.

Additionally, the threshold used for peak-picking (0.3 by default) and a gpu device ID (if gpu is available) can be set using the below arguments:
```
--threshold <value> --gpu_id <ID>
```
