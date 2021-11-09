from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from dataset import nomakeup_imgs, makeup_imgs, nomakeup_segs, makeup_segs


def make_cdfs(imgs, segs):
    cdfs = np.zeros((imgs.shape[0], 3, 3, 256))

    for i in range(imgs.shape[0]):
        for j in range(3):
            points = np.nonzero(segs[i, j])
            n = len(points)
            colors = imgs[(i, *points)]
            for k in range(3):
                for l in range(cdfs.shape[3]):
                    cdfs[i, j, k, l] = (colors[:, k]<l/255.0).sum()/n
    return cdfs

try:
    makeup_cdfs = np.load('makeup_cdfs.npy')
    nomakeup_cdfs = np.load('nomakeup_cdfs.npy')
except:
    makeup_cdfs = make_cdfs(makeup_imgs, makeup_segs)
    nomakeup_cdfs = make_cdfs(nomakeup_imgs, nomakeup_segs)
    np.save('makeup_cdfs.npy', makeup_cdfs)
    np.save('nomakeup_cdfs.npy', nomakeup_cdfs)

def match_histograms(img, cdf, target_cdf, mask):
    transformed = np.zeros_like(img)
    points = np.nonzero(mask)
    colors = (img[points]*255).astype(np.int)
    # transformed_colors = transformed[points]
    # print(cdf.shape)

    for i in range(3):
        t = cdf[i][colors[:, i]][np.newaxis, ...]
        t2 = ((t-target_cdf[i][..., np.newaxis])**2).argmin(axis=0)
        transformed[(*points, i)] = t2.astype(np.float32)/255.0

    return transformed

    # for c, transformed in zip(colors, transformed_colors):
    #     for i in range(3):
    #         t = cdf[(c[i]*255)]



# plt.plot(makeup_cdfs[0, 0, 0])
# plt.show()

# fig, ax = plt.subplots(8, 11)
# for i in range(8):
#     for j in range(11):
#         ax[i, j].get_xaxis().set_visible(False)
#         ax[i, j].get_yaxis().set_visible(False)

#     ax[i, 0].imshow(nomakeup_imgs[i])
#     ax[i, 1].imshow(makeup_imgs[i])

#     for j in range(3):
      
#         ax[i, j+2].imshow(nomakeup_imgs[i] *
#                           (nomakeup_segs[i, j][..., np.newaxis]))
#     for j in range(3):

#         ax[i, j+5].imshow(makeup_imgs[i] *
#                           (makeup_segs[i, j][..., np.newaxis]))

#     for j in range(3):
#         image = nomakeup_imgs[i] * (nomakeup_segs[i, j][..., np.newaxis])
#         target_image = match_histograms(image, nomakeup_cdfs[i, j], makeup_cdfs[i, j], nomakeup_segs[i, j])

        
#         ax[i, j+8].imshow(target_image)

# plt.show()

