import tensorflow as tf


def upsample_block(filters, kernel_size):
    initializer = tf.random_normal_initializer(0., 0.02)
    ct = tf.keras.layers.Conv2DTranspose(
        filters, kernel_size, strides=2, padding='same', use_bias=False, kernel_initializer=initializer)
    return tf.keras.Sequential([
        ct,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
    ])


base_model = tf.keras.applications.MobileNetV2(
    input_shape=[256, 256, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(
    name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input,
                            outputs=base_model_outputs)

down_stack.trainable = True

up_stack = [
    upsample_block(512, 3),
    upsample_block(256, 3), 
    upsample_block(128, 3),
    upsample_block(64, 3),  
]

def unet_model() -> tf.keras.Model:
    src_image = tf.keras.layers.Input(shape=(256, 256, 3))
    ref_image = tf.keras.layers.Input(shape=(256, 256, 3))

    # Downsampling through the model
    src_skips = down_stack(src_image)
    ref_skips = down_stack(ref_image)
    x = tf.concat([src_skips[-1], ref_skips[-1]], axis=-1) 

    src_skips = reversed(src_skips[:-1])
    ref_skips = reversed(ref_skips[:-1])

    # Upsampling and establishing the skip connections
    for up, src_skip, ref_skip in zip(up_stack, src_skips, ref_skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, src_skip, ref_skip])

    last = tf.keras.layers.Conv2DTranspose(
        filters=3, kernel_size=3, strides=2,
        padding='same', activation='sigmoid', name='last_A')

    last2 = tf.keras.layers.Conv2DTranspose(
        filters=3, kernel_size=3, strides=2,
        padding='same', activation='sigmoid', name='last_B')

    src_B = last(x)
    ref_A = last2(x)

    return tf.keras.Model(inputs=[src_image, ref_image], outputs=[src_B, ref_A])