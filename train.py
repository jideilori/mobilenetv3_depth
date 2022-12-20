# Compile & Train
import tensorflow as tf
from loss import depth_loss_function
import os
from load_data import DataLoader
from model import MobileNetv3_model

# Parameters
batch_size     = 16
learning_rate  = 0.0001
epochs         = 1


dl = DataLoader()
train_generator = dl.get_batched_dataset(batch_size)

print('Data loader ready.')



optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)

MobileNetv3_model.compile(loss=depth_loss_function, optimizer=optimizer)

# Create checkpoint callback
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

MobileNetv3_model.fit(train_generator, epochs=5, steps_per_epoch=dl.length//batch_size, callbacks=[cp_callback])


MobileNetv3_model.save('model_full.h5')
MobileNetv3_model = tf.keras.models.load_model('model_full.h5', compile=False)

# Converting a tf.Keras model to a TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_keras_model(MobileNetv3_model)
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile('mobilenetv3_model.tflite', 'wb') as f:
  f.write(tflite_model)