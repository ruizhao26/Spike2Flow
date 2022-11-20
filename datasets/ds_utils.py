import numpy as np
import os
import h5py
import cv2
import os.path as osp

def make_dir(path):
    if not osp.exists(path):
        os.makedirs(path)
    return

def RawToSpike(video_seq, h, w, flipud=True):
    video_seq = np.array(video_seq).astype(np.uint8)
    img_size = h*w
    img_num = len(video_seq)//(img_size//8)
    SpikeMatrix = np.zeros([img_num, h, w], np.uint8)
    pix_id = np.arange(0,h*w)
    pix_id = np.reshape(pix_id, (h, w))
    comparator = np.left_shift(1, np.mod(pix_id, 8))
    byte_id = pix_id // 8

    for img_id in np.arange(img_num):
        id_start = img_id*img_size//8
        id_end = id_start + img_size//8
        cur_info = video_seq[id_start:id_end]
        data = cur_info[byte_id]
        result = np.bitwise_and(data, comparator)
        if flipud:
            SpikeMatrix[img_id, :, :] = np.flipud((result == comparator))
        else:
            SpikeMatrix[img_id, :, :] = (result == comparator)

    return SpikeMatrix

def save_to_h5(SpikeMatrix, h5path):
    f = h5py.File(h5path, 'w')
    f['raw_spike'] = SpikeMatrix
    f.close()

def dat_to_h5(dat_path, h5path, size=[436, 1024]):
    f = open(dat_path, 'rb')
    video_seq = f.read()
    video_seq = np.frombuffer(video_seq, 'b')
    sp_mat = RawToSpike(video_seq, size[0], size[1])
    save_to_h5(sp_mat, h5path)

def dat_to_spmat(dat_path,  size=[436, 1024]):
    f = open(dat_path, 'rb')
    video_seq = f.read()
    video_seq = np.frombuffer(video_seq, 'b')
    sp_mat = RawToSpike(video_seq, size[0], size[1])
    return sp_mat


############################################################################
## General Read Function

def read_gen(file_name):
    ext = osp.splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        # return Image.open(file_name)
        return cv2.imread(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    return []


def im2gray(im):
    # im = np.array(im).astype(np.float32)[..., :3] / 255.
    # return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = im.astype(np.float32) / 255.
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = np.expand_dims(im, axis=0)
    return im

def im_color(im):
    im = im.astype(np.float32) / 255.
    im = im.transpose([2, 0, 1])
    return im

TAG_CHAR = np.array([202021.25], np.float32)
def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))