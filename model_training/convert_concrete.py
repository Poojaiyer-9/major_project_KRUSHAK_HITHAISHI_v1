"""
convert_concrete.py — TFLite conversion via concrete function
Workaround for MobileNetV3Small + TF 2.16 + Keras 3 LLVM bug.
Uses tf.function concrete function tracing instead of SavedModel converter.
"""
import os, shutil, tempfile
import numpy as np
import tensorflow as tf

MODEL_PATH  = r'C:\Users\Admis\Downloads\major_project_krushak_hithaishi\model_training\model.keras'
OUTPUT_PATH = r'C:\Users\Admis\Downloads\major_project_krushak_hithaishi\model_training\krushak.tflite'
IMAGE_SIZE  = 224

print('[1/4] Loading model.keras ...')
model = tf.keras.models.load_model(MODEL_PATH)
print('      Inputs :', [i.name for i in model.inputs])
print('      Outputs:', [o.name for o in model.outputs])

# Trace a concrete function with fixed batch=1 shapes
print('[2/4] Tracing concrete function ...')

@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, IMAGE_SIZE, IMAGE_SIZE, 3], dtype=tf.float32, name='image'),
    tf.TensorSpec(shape=[1, 4],                         dtype=tf.float32, name='weather'),
])
def predict_fn(image, weather):
    return model({'image': image, 'weather': weather}, training=False)

concrete_func = predict_fn.get_concrete_function()
print('      OK — concrete function traced')

# Convert from concrete function
print('[3/4] Converting to TFLite (float16 quantization) ...')
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_func], predict_fn
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

try:
    tflite_model = converter.convert()
except Exception as e:
    print('[WARN] float16 conversion failed:', e)
    print('[WARN] Retrying without quantization ...')
    converter2 = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_func], predict_fn
    )
    tflite_model = converter2.convert()

with open(OUTPUT_PATH, 'wb') as f:
    f.write(tflite_model)

size_kb = os.path.getsize(OUTPUT_PATH) / 1024
print('[4/4] Saved', OUTPUT_PATH, '(%.1f KB)' % size_kb)

# Sanity check with a dummy inference
print('[OK] Running sanity inference ...')
interp = tf.lite.Interpreter(model_path=OUTPUT_PATH)
interp.allocate_tensors()
in_details  = interp.get_input_details()
out_details = interp.get_output_details()

for d in in_details:
    print('  Input :', d['name'], d['shape'].tolist(), d['dtype'])
for d in out_details:
    print('  Output:', d['shape'].tolist(), d['dtype'])

# Run one dummy prediction
dummy_img = np.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.float32)
dummy_wx  = np.zeros([1, 4], dtype=np.float32)

img_in = next(d for d in in_details if len(d['shape']) == 4)
wx_in  = next(d for d in in_details if d['shape'][-1] == 4)
interp.set_tensor(img_in['index'], dummy_img)
interp.set_tensor(wx_in['index'],  dummy_wx)
interp.invoke()

for d in out_details:
    out = interp.get_tensor(d['index'])
    print('  Predicted class:', int(np.argmax(out)), '  shape:', out.shape)

print('[DONE] TFLite model is working correctly')
