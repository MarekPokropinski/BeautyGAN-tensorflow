import matplotlib.pyplot as plt
from histogram_matching import makeup_imgs, nomakeup_imgs, makeup_cdfs, nomakeup_cdfs, makeup_segs, nomakeup_segs
from models import BeautyGAN
import numpy as np

fig, ax = plt.subplots(4, 4)
idx1 = [1, 2, 5]
idx2= [0, 5, 7]

beautyGAN = BeautyGAN([256, 256])
beautyGAN.discriminatorA.predict(makeup_imgs[:1])
beautyGAN.discriminatorB.predict(makeup_imgs[:1])
beautyGAN.load('model')

predictions = {}

for i in idx1:
    for j in idx2:
        src = nomakeup_imgs[-10:][i][np.newaxis, ...]
        ref = makeup_imgs[-10:][j][np.newaxis, ...]
        [pred], _ = beautyGAN.generator.predict([src, ref])
        predictions[(i, j)] = pred

for i, id1 in enumerate(idx1):
    ax[0, i+1].imshow(nomakeup_imgs[-10:][id1])
    for j, id2 in enumerate(idx2):
        ax[j+1, i+1].imshow(predictions[id1, id2])

for i, id1 in enumerate(idx2):
    ax[i+1, 0].imshow(makeup_imgs[-10:][id1])

for i in range(4):
    for j in range(4):
        ax[j, i].get_xaxis().set_visible(False)
        ax[j, i].get_yaxis().set_visible(False)
        ax[j, i].axis('off')

plt.show()