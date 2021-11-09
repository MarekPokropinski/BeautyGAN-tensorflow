import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from image_loss import PerceptualLoss
from unet import unet_model
from functools import partial
import os
from histogram_matching import match_histograms, make_cdfs

IMAGE_SIZE = (256, 256)

perceptualLoss = PerceptualLoss((*IMAGE_SIZE, 3))


def build_generator(image_size):
    return unet_model()


# def build_discriminator(image_size):
#     model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=(*image_size, 3), pooling='max')
#     model.trainable = True
#     out = tf.keras.layers.Dense(1, activation='sigmoid')(model.output)
#     return tf.keras.models.Model(model.input, out)

def build_discriminator(image_size):
    layers = [
        tf.keras.layers.Conv2D(64, kernel_size=4, strides=2),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Conv2D(128, kernel_size=4, strides=2),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Conv2D(256, kernel_size=4, strides=2),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Conv2D(512, kernel_size=4, strides=1),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, use_bias=False),
        tf.keras.layers.Activation(tf.keras.activations.sigmoid)
    ]
    layers = [tfa.layers.SpectralNormalization(l) if type(
        l) == tf.keras.layers.Conv2D else l for l in layers]
    model = tf.keras.models.Sequential(
        [*layers, tf.keras.layers.GlobalAveragePooling2D()])

    return model


class BeautyGAN:
    def __init__(self, image_size) -> None:
        self.image_size = image_size
        self.generator = build_generator(image_size)
        self.discriminatorA = build_discriminator(image_size)
        self.discriminatorB = build_discriminator(image_size)
        self.adversarial_loss_A = tf.function(
            partial(BeautyGAN.adversarial_loss, discriminator=self.discriminatorA))
        self.adversarial_loss_B = tf.function(
            partial(BeautyGAN.adversarial_loss, discriminator=self.discriminatorB))

        self.optim = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.mae = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    @tf.function
    def cycle_consistency_loss(self, src_reconstruction, src_ground_truth, ref_reconstruction, ref_ground_truth):
        # dist1 = tf.sqrt(tf.reduce_sum(tf.square(src_reconstruction - src_ground_truth), [1, 2, 3]))
        # dist2 = tf.sqrt(tf.reduce_sum(tf.square(ref_reconstruction - ref_ground_truth), [1, 2, 3]))
        # dist1 = tf.reduce_mean(tf.square(src_reconstruction - src_ground_truth))
        # dist2 = tf.reduce_mean(tf.square(ref_reconstruction - ref_ground_truth))
        dist1 = tf.reduce_mean(self.mae(src_reconstruction, src_ground_truth))
        dist2 = tf.reduce_mean(self.mae(ref_reconstruction, ref_ground_truth))
        return dist1+dist2

    @staticmethod
    def adversarial_loss(true, fake, discriminator):
        eps = 0.1
        D_fake = discriminator(fake)
        if true is None:
            return tf.reduce_mean((D_fake-1.0)**2)
        D_true = discriminator(true)
        return 0.5 * (tf.reduce_mean((D_true-1.0)**2) + tf.reduce_mean(D_fake**2))
        # return 0.5 * (tf.reduce_mean(tf.math.log(1-D_fake)) + tf.reduce_mean(tf.math.log(D_true)))
        # return 0.5 * (tf.reduce_mean((D_fake-1.0-eps)**2-eps**2) + tf.reduce_mean((D_true-eps)**2-eps**2))

    @tf.function
    def makeup_loss(self, fake, face_matched, lips_matched, eyes_matched, masks):
        masks = tf.cast(masks, tf.float32)
        l_face = self.mae(fake, face_matched, sample_weight=masks[:, 0][..., tf.newaxis])
        l_lips = self.mae(fake, lips_matched, sample_weight=masks[:, 1][..., tf.newaxis])
        l_eyes = self.mae(fake, eyes_matched, sample_weight=masks[:, 2][..., tf.newaxis])
        return 0.1 * tf.reduce_mean(l_face) + 1.0 * tf.reduce_mean(l_lips) + 1.0 * tf.reduce_mean(l_eyes)

    @tf.function
    def _train_on_batch(self, source_batch, ref_batch):
        fake_A, fake_B = self.generator([source_batch, ref_batch])
        with tf.GradientTape() as tape:
            # discriminators maximize adversarial loss
            D_A_loss = self.adversarial_loss_A(source_batch, fake_B)
            D_B_loss = self.adversarial_loss_B(ref_batch, fake_A)
            D_loss = 0.5*(D_A_loss + D_B_loss)

        vars = tape.watched_variables()
        grads = tape.gradient(D_loss, vars)
        self.optim.apply_gradients(zip(grads, vars))
        # grads = tape.gradient(D_A_loss, self.discriminatorA.trainable_variables)
        # self.optim.apply_gradients(zip(grads, self.discriminatorA.trainable_variables))

        # grads = tape.gradient(D_B_loss, self.discriminatorB.trainable_variables)
        # self.optim.apply_gradients(zip(grads, self.discriminatorB.trainable_variables))

        face_matched_A, lips_matched_A, eyes_matched_A, src_masks, face_matched_B, lips_matched_B, eyes_matched_B, ref_masks = tf.py_function(
            self.histogram_matching, inp=[fake_A, fake_B], Tout=[tf.float32, tf.float32, tf.float32, tf.bool, tf.float32, tf.float32, tf.float32, tf.bool])

        with tf.GradientTape() as tape:
            fake_A, fake_B = self.generator([source_batch, ref_batch])

            D_A_loss2 = self.adversarial_loss_A(None, fake_B)
            D_B_loss2 = self.adversarial_loss_B(None, fake_A)

            adversarial_loss = 1*0.5*(D_A_loss2 + D_B_loss2)

            perceptual_loss = 0.005 * \
                (perceptualLoss(source_batch, fake_A) +
                 perceptualLoss(ref_batch, fake_B))

            makeup_loss_A = self.makeup_loss(
                fake_A, face_matched_A, lips_matched_A, eyes_matched_A, src_masks)
            makeup_loss_B = self.makeup_loss(
                fake_B, face_matched_B, lips_matched_B, eyes_matched_B, ref_masks)
            makeup_loss = 50*0.5*(makeup_loss_A+makeup_loss_B)

            rec_B, rec_A = self.generator([fake_B, fake_A])

            cycle_loss = 10 * \
                self.cycle_consistency_loss(
                    rec_A, source_batch, rec_B, ref_batch)
            total_loss = cycle_loss + perceptual_loss + adversarial_loss + makeup_loss

        grads = tape.gradient(total_loss, self.generator.trainable_variables)
        self.optim.apply_gradients(
            zip(grads, self.generator.trainable_variables))

        return D_A_loss, D_B_loss, adversarial_loss, perceptual_loss, cycle_loss, makeup_loss, total_loss

    def histogram_matching(self, fake_A, fake_B):
        fake_A = fake_A.numpy()
        fake_B = fake_B.numpy()
        source_batch, ref_batch, src_masks, ref_masks, source_cdfs, ref_cdfs = self.current_args
        face_matched_A = np.empty_like(fake_A)
        lips_matched_A = np.empty_like(fake_A)
        eyes_matched_A = np.empty_like(fake_A)

        face_matched_B = np.empty_like(fake_B)
        lips_matched_B = np.empty_like(fake_B)
        eyes_matched_B = np.empty_like(fake_B)

        fake_cdfs_A = make_cdfs(fake_A, src_masks)
        fake_cdfs_B = make_cdfs(fake_B, ref_masks)

        for i in range(fake_A.shape[0]):
            face_matched_A[i] = match_histograms(
                fake_A[i], fake_cdfs_A[i, 0], ref_cdfs[i, 0], src_masks[i, 0])
            lips_matched_A[i] = match_histograms(
                fake_A[i], fake_cdfs_A[i, 1], ref_cdfs[i, 1], src_masks[i, 1])
            eyes_matched_A[i] = match_histograms(
                fake_A[i], fake_cdfs_A[i, 2], ref_cdfs[i, 2], src_masks[i, 2])

            face_matched_B[i] = match_histograms(
                fake_B[i], fake_cdfs_B[i, 0], source_cdfs[i, 0], ref_masks[i, 0])
            lips_matched_B[i] = match_histograms(
                fake_B[i], fake_cdfs_B[i, 1], source_cdfs[i, 1], ref_masks[i, 1])
            eyes_matched_B[i] = match_histograms(
                fake_B[i], fake_cdfs_B[i, 2], source_cdfs[i, 2], ref_masks[i, 2])

        return face_matched_A, lips_matched_A, eyes_matched_A, src_masks, face_matched_B, lips_matched_B, eyes_matched_B, ref_masks

    def train_on_batch(self, source_batch, ref_batch, src_masks, ref_masks, source_cdfs, ref_cdfs):
        self.current_args = (source_batch, ref_batch,
                             src_masks, ref_masks, source_cdfs, ref_cdfs)
        return [x.numpy() for x in self._train_on_batch(source_batch, ref_batch)]

    def predict(self, source_batch, ref_batch):
        fake_A, fake_B = self.generator([source_batch, ref_batch])
        rec_B, rec_A = self.generator([fake_B, fake_A])
        return fake_A.numpy(), fake_B.numpy(), rec_A.numpy(), rec_B.numpy()

    def save(self, filename):
        self.discriminatorA.save_weights(
            os.path.join(filename, 'discriminatorA' + '.h5'))
        self.discriminatorB.save_weights(
            os.path.join(filename, 'discriminatorB' + '.h5'))
        self.generator.save_weights(
            os.path.join(filename, 'generator' + '.h5'))
        # with open(os.path.join(filename, 'optimizer.json'), 'w') as f:
        #     json.dump(self.optim.get_config(), f)

    def load(self, filename):
        self.discriminatorA.load_weights(
            os.path.join(filename, 'discriminatorA' + '.h5'))
        self.discriminatorB.load_weights(
            os.path.join(filename, 'discriminatorB' + '.h5'))
        self.generator.load_weights(
            os.path.join(filename, 'generator' + '.h5'))
