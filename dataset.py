import os
import numpy as np
from imageio import imread
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod

DATASET_PATH = r"C:\Programming\datasets\makeup_dataset\all"

def rebound_box(left_eye, right_eye, face):
    mask = np.zeros(face.shape, dtype='bool')
    idx = np.argwhere(left_eye)
    left = idx[:, 0].min()-10
    right = idx[:, 0].max()+10
    bottom = idx[:, 1].min()-10
    top = idx[:, 1].max()+10
    mask[left:right, bottom:top] = face[left:right, bottom:top]

    idx = np.argwhere(right_eye)
    left = idx[:, 0].min()-10
    right = idx[:, 0].max()+10
    bottom = idx[:, 1].min()-10
    top = idx[:, 1].max()+10
    mask[left:right, bottom:top] = face[left:right, bottom:top]
    return mask

try:
    nomakeup_imgs = np.load('nomakeup_imgs.npy')
    nomakeup_segs = np.load('nomakeup_segs.npy')
    makeup_imgs = np.load('makeup_imgs.npy')
    makeup_segs = np.load('makeup_segs.npy')
except:
    makeup_files = [p for p in os.listdir(os.path.join(DATASET_PATH, 'images', 'makeup'))]
    nomakeup_files = [p for p in os.listdir(os.path.join(DATASET_PATH, 'images', 'non-makeup'))]

    makeup_image_files = [os.path.join(DATASET_PATH, 'images', 'makeup', p) for p in makeup_files]
    nomakeup_image_files = [os.path.join(DATASET_PATH, 'images', 'non-makeup', p) for p in nomakeup_files]

    makeup_segs_files = [os.path.join(DATASET_PATH, 'segs', 'makeup', p) for p in makeup_files]
    nomakeup_segs_files = [os.path.join(DATASET_PATH, 'segs', 'non-makup', p) for p in nomakeup_files]

    makeup_imgs = np.empty((len(makeup_image_files), 256, 256, 3), dtype='float32')
    nomakeup_imgs = np.empty((len(nomakeup_image_files), 256, 256, 3), dtype='float32')

    _makeup_segs = np.empty((len(makeup_image_files), 256, 256, 1), dtype='uint8')
    _nomakeup_segs = np.empty((len(nomakeup_image_files), 256, 256, 1), dtype='uint8')

    makeup_segs = np.empty((len(makeup_image_files), 256, 256, 3), dtype='bool')
    nomakeup_segs = np.empty((len(nomakeup_image_files), 256, 256, 3), dtype='bool')

    for i, path in enumerate(makeup_image_files):
        makeup_imgs[i] = tf.image.resize(imread(path)/255.0, [256, 256])

    for i, path in enumerate(nomakeup_image_files):
        nomakeup_imgs[i] = tf.image.resize(imread(path)/255.0, [256, 256])

    for i, path in enumerate(makeup_segs_files):
        _makeup_segs[i] = tf.image.resize(imread(path)[..., tf.newaxis], [256, 256], method=ResizeMethod.NEAREST_NEIGHBOR)

    for i, path in enumerate(nomakeup_segs_files):
        _nomakeup_segs[i] = tf.image.resize(imread(path)[..., tf.newaxis], [256, 256], method=ResizeMethod.NEAREST_NEIGHBOR)




    makeup_segs[_makeup_segs[:, :, :, 0]==1, 0] = True
    makeup_segs[_makeup_segs[:, :, :, 0]==6, 0] = True
    makeup_segs[_makeup_segs[:, :, :, 0]==13, 0] = True

    makeup_segs[_makeup_segs[:, :, :, 0]==7, 1] = True
    makeup_segs[_makeup_segs[:, :, :, 0]==9, 1] = True

    indices = []
    for i in range(_makeup_segs.shape[0]):
        try:
            makeup_segs[i, :, :, 2] = rebound_box(_makeup_segs[i, :, :, 0]==4, _makeup_segs[i, :, :, 0]==5, makeup_segs[i, :, :, 0])
            indices.append(i)
        except:
            pass

    makeup_segs = makeup_segs.transpose([0, 3, 1, 2])[indices]
    makeup_imgs = makeup_imgs[indices]

    nomakeup_segs[_nomakeup_segs[:, :, :, 0]==1, 0] = True
    nomakeup_segs[_nomakeup_segs[:, :, :, 0]==6, 0] = True
    nomakeup_segs[_nomakeup_segs[:, :, :, 0]==13, 0] = True

    nomakeup_segs[_nomakeup_segs[:, :, :, 0]==7, 1] = True
    nomakeup_segs[_nomakeup_segs[:, :, :, 0]==9, 1] = True

    indices = []
    for i in range(_nomakeup_segs.shape[0]):
        try:
            nomakeup_segs[i, :, :, 2] = rebound_box(_nomakeup_segs[i, :, :, 0]==4, _nomakeup_segs[i, :, :, 0]==5, nomakeup_segs[i, :, :, 0])
            indices.append(i)
        except:
            pass

    nomakeup_segs = nomakeup_segs.transpose([0, 3, 1, 2])[indices]
    nomakeup_imgs = nomakeup_imgs[indices]

    np.save('nomakeup_imgs.npy', nomakeup_imgs)
    np.save('nomakeup_segs.npy', nomakeup_segs)
    np.save('makeup_imgs.npy', makeup_imgs)
    np.save('makeup_segs.npy', makeup_segs)

    del _makeup_segs
    del _nomakeup_segs