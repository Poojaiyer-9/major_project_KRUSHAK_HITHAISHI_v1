"""
convert_savedmodel.py — TFLite conversion via SavedModel route
Workaround for Keras3 + TF2.16 direct TFLite conversion bug.
"""
import os, shutil, tempfile
import tensorflow as tf

MODEL_PATH   = r'C:\Users\Admis\Downloads\major_project_krushak_hithaishi\model_training\model.keras'
OUTPUT_PATH  = r'C:\Users\Admis\Downloads\major_project_krushak_hithaishi\model_training\krushak.tflite'

print('[1/4] Loading model.keras ...')
model = tf.keras.models.load_model(MODEL_PATH)
print('      Inputs :', [i.name for i in model.inputs])
print('      Outputs:', [o.name for o in model.outputs])

# Step 1: export to SavedModel first (workaround for Keras3/TFLite LLVM bug)
saved_model_dir = os.path.join(tempfile.gettempdir(), 'krushak_saved')
if os.path.exists(saved_model_dir):
    shutil.rmtree(saved_model_dir)
print('[2/4] Saving as SavedModel to', saved_model_dir)
tf.saved_model.save(model, saved_model_dir)

# Step 2: convert from SavedModel with float16 quantization
print('[3/4] Converting SavedModel -> TFLite (float16) ...')
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()
with open(OUTPUT_PATH, 'wb') as f:
    f.write(tflite_model)

size_kb = os.path.getsize(OUTPUT_PATH) / 1024
print('[4/4] Saved', OUTPUT_PATH, '(%.1f KB)' % size_kb)

# Sanity check
interp = tf.lite.Interpreter(model_path=OUTPUT_PATH)
interp.allocate_tensors()
in_details  = interp.get_input_details()
out_details = interp.get_output_details()
for d in in_details:
    print('  Input :', d['name'], d['shape'].tolist(), d['dtype'])
for d in out_details:
    print('  Output:', d['name'], d['shape'].tolist(), d['dtype'])

shutil.rmtree(saved_model_dir, ignore_errors=True)
print('[DONE] TFLite conversion successful')
