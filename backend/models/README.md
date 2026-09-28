Place the trained `krushak.tflite` model here (output of `model_training/convert_tflite.py`).

Also required in this folder:
- `disease_labels.txt`  (38 class names, one per line)
- `severity_labels.txt` (3 class names: LOW / MEDIUM / HIGH)

The backend loads `krushak.tflite` lazily on the first `/detect` request. If the file
is missing, the `/detect` endpoint returns a clearly-labelled demo result so the rest
of the pipeline (translation, voice, heatmap, shop lookup) can still be exercised.
