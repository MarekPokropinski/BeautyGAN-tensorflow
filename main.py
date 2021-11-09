import tensorflow as tf
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from models import BeautyGAN
from histogram_matching import makeup_imgs, nomakeup_imgs, makeup_cdfs, nomakeup_cdfs, makeup_segs, nomakeup_segs
import os
import random

BATCH_SIZE = 4


makeup_indices = list(range(makeup_imgs.shape[0]-10))
nomakeup_indices = list(range(nomakeup_imgs.shape[0]-10))

beautyGAN = BeautyGAN([256, 256])
beautyGAN.discriminatorA.predict(makeup_imgs[:1])
beautyGAN.discriminatorB.predict(makeup_imgs[:1])
# beautyGAN.load('model')
# down_stack.trainable = True
src_test_imgs = np.concatenate([nomakeup_imgs[:3], nomakeup_imgs[-3:]], axis=0)
ref_test_imgs = np.concatenate([makeup_imgs[:3], makeup_imgs[-3:]], axis=0)

for step in range(1000000):
    idx = random.sample(makeup_indices, BATCH_SIZE)
    idx2 = random.sample(nomakeup_indices, BATCH_SIZE)

    train_data = beautyGAN.train_on_batch(nomakeup_imgs[idx2], makeup_imgs[idx], nomakeup_segs[idx2], makeup_segs[idx], nomakeup_cdfs[idx2], makeup_cdfs[idx])
    if step%10==0:
        fake_A, fake_B, rec_A, rec_B = beautyGAN.predict(src_test_imgs, ref_test_imgs)
        fig, ax = plt.subplots(6, 6)
        for i in range(6):
            ax[0, i].imshow(src_test_imgs[i])
            ax[1, i].imshow(ref_test_imgs[i])
            ax[2, i].imshow(fake_A[i])
            ax[3, i].imshow(fake_B[i])
            ax[4, i].imshow(rec_A[i])
            ax[5, i].imshow(rec_B[i])

            for j in range(6):
                ax[j, i].get_xaxis().set_visible(False)
                ax[j, i].get_yaxis().set_visible(False)

        plt.savefig(f'out/{step}.png')
        plt.close()
    
    if step%100==0:
        beautyGAN.save('model')

    print(train_data)