from skimage import io, transform
import glob
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import math

INPUT_DATA = 'picture_3/'


def radia_transform(im, w_ratio, h_ratio):
    shape = im.shape
    new_im = np.zeros(shape)
    print(shape)
    width = shape[1]
    height = shape[0]
    w = int(w_ratio*width)
    h = int(h_ratio*height)
    print('w', w)
    print('h', h)
    lens = len(shape)
    for i in range(0, height):
        theta = 2*np.pi*i/height
        for a in range(0, width):
            x = int(a * math.cos(theta))
            y = int(a * math.sin(theta))
            new_x = int(w+x)
            new_y = int(h-y)
            #print(h.dtype)
            if 0 <= new_x < width:
                if 0 <= new_y < height:
                    if lens == 3:
                        # new_im[i, a, 0] = (im[new_y, new_x, 0]-127.5)/128
                        # new_im[i, a, 1] = (im[new_y, new_x, 1]-127.5)/128
                        # new_im[i, a, 2] = (im[new_y, new_x, 2]-127.5)/128
                        new_im[a, i, 0] = im[new_y, new_x, 0]
                        new_im[a, i, 1] = im[new_y, new_x, 1]
                        new_im[a, i, 2] = im[new_y, new_x, 2]
                    else:
                        new_im[a, i] = (im[new_y, new_x]-127.5)/128
                        new_im[a, i] = (im[new_y, new_x]-127.5)/128
                        new_im[a, i] = (im[new_y, new_x]-127.5)/128
    return new_im

def create_image_lists():
    # 得到的所有图片都存在result这个字典(dictionary)里。
    # 这个字典的key为类别的名称，value也是一个字典，字典里存储了所有的图片名称。
    result = {}
    # 获取当前目录下所有的子目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # 得到的第一个目录是当前目录，不需要考虑
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取当前目录下所有的有效图片文件。
        # extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        # for extension in extensions:
        file_glob = os.path.join(INPUT_DATA, dir_name, '*.jpg')
        file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue
        print(len(file_list))
        # 通过目录名获取类别的名称。
        label_name = dir_name.lower()
        base_name = []
        for file_name in file_list:
            base_name.append(file_name)
            # 将当前类别的数据放入结果字典。

        result[label_name] = base_name
    # 返回整理好的所有数据
    return result

def read_img(im):
    img = io.imread(im)
    img = radia_transform(img, 0.5, 0.5)
    img = transform.resize(img, (299, 299))
    return np.asarray(img, np.float32)

dir_path = create_image_lists()


with gfile.FastGFile('inception_v4.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    # 加载读取的Inception-v3。
bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=['InceptionV4/Logits/PreLogitsFlatten/Reshape:0', 'InputImage:0'])
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    i = 0
    for i in dir_path:
            for j in dir_path[i]:
                data = read_img(j)
                bottleneck_values = sess.run(bottleneck_tensor, {jpeg_data_tensor: np.asarray([data])})
                bottleneck_values = np.squeeze(bottleneck_values)
                bottleneck_string = ','.join(str(x) for x in bottleneck_values)
                with open((os.path.join('./tmp/tensor', i, os.path.basename(j))+'_radial(0.5_0.5).txt'), 'w') as bottleneck_file:
                    bottleneck_file.write(bottleneck_string)
