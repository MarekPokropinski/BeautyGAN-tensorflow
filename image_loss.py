import tensorflow as tf

content_layers = [
    'block4_conv1'
]


def vgg(input_shape):
    input = tf.keras.layers.Input(shape=input_shape)
    net = tf.keras.applications.vgg16.VGG16(
        input_tensor=input,
        weights='imagenet',
        include_top=False
    )

    output = None

    for layer in net.layers:
        if layer.name in content_layers:
            output = layer.output
            layer.trainable = False
            break

    return tf.keras.models.Model(input, output)


class PerceptualLoss:
    def __init__(self, image_shape):
        self.image_shape = image_shape
        self.model = vgg(image_shape)

    def calculate_loss(self, original_image, image_tensor):
        content_output = self.model(original_image)
        prediction = self.model(image_tensor)        
        content_loss = tf.math.reduce_mean(tf.math.square(content_output-prediction))
        return content_loss

    def __call__(self, original_image, image_tensor):
        original_image = original_image*255
        image_tensor = image_tensor*255
        return self.calculate_loss(original_image, image_tensor)
