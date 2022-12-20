# Compile & Train
import tensorflow
from loss import depth_loss_function
import os
from load_data import DataLoader
from model import MobileNetv3_model

# Parameters
batch_size     = 16
learning_rate  = 0.0001
epochs         = 10


dl = DataLoader()
train_generator = dl.get_batched_dataset(batch_size)

print('Data loader ready.')



optimizer = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)

MobileNetv3_model.compile(loss=depth_loss_function, optimizer=optimizer)

# Create checkpoint callback
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

MobileNetv3_model.fit(train_generator, epochs=5, steps_per_epoch=dl.length//batch_size, callbacks=[cp_callback])