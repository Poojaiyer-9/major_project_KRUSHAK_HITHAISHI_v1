Train the model for real, right now, on this machine. Steps:

1. Run `nvidia-smi` and `python -c "import tensorflow as tf; 
   print(tf.config.list_physical_devices())"` and tell me whether a 
   GPU is available. If not, use CPU-friendly settings below.

2. Get data: download PlantVillage (Kaggle CLI or tensorflow-datasets, 
   whichever works without me logging in; if it needs my Kaggle API 
   key, stop and ask me). Keep ONLY the 10 tomato classes.

3. Run data_prep.py with this fix: synthetic weather values must be 
   noisy and overlap heavily between classes (weak correlation with 
   disease), so the weather branch can't just read the label. Add a 
   code comment explaining why.

4. Train with train_model.py: pretrained ImageNet MobileNetV3Small, 
   frozen backbone for the first 5 epochs, then unfreeze the top 
   layers for a few more. Image size 224. If CPU-only: use a 
   subsample of ~300 images per class and 3-5 epochs so it finishes 
   in under 60 minutes.

5. Run convert_tflite.py and evaluate.py. Also train the image-only 
   baseline with the same settings for a fair comparison.

6. Put krushak.tflite, disease_labels.txt and severity_labels.txt in 
   backend/models/, restart the backend, and call /detect with a real 
   tomato leaf image. Confirm the response has no "demo": true.

7. Print the real numbers from evaluate.py: test accuracy for 
   multimodal vs image-only, model size in KB, average inference 
   latency in ms over 50 runs.

Rules: never estimate or make up any metric. Everything reported must 
come from output you actually ran. If training fails or the accuracy 
looks suspiciously high, say so and investigate instead of hiding it.