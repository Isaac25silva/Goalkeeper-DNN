#coding: utf-8
#!/usr/bin/env python2



"""
Classify an image using a model archive file
"""

import base64
import h5py
import logging
import PIL.Image
import argparse
import os
import tempfile
import time
from math import log,exp,tan,radians
import thread
import imutils


import ctypes
import os
import cv2
#import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
from servo import Servo
import sys

""" Initiate the path to blackboard (Shared Memory)"""
sys.path.append('../../Blackboard/src/')
"""Import the library Shared memory """
from SharedMemory import SharedMemory 
""" Treatment exception: Try to import configparser from python. Write and Read from config.ini file"""
try:
    """There are differences in versions of the config parser
    For versions > 3.0 """
    from ConfigParser import ConfigParser
except ImportError:
    """For versions < 3.0 """
    from ConfigParser import ConfigParser 



""" Instantiate bkb as a shared memory """
bkb = SharedMemory()
""" Config is a new configparser """
config = ConfigParser()
""" Path for the file config.ini:"""
config.read('../../Control/Data/config.ini')
""" Mem_key is for all processes to know where the blackboard is. It is robot number times 100"""
#mem_key = int(config.get('Communication', 'no_player_robofei'))*100
mem_key = int(3)*100
"""Memory constructor in mem_key"""
Mem = bkb.shd_constructor(mem_key)


try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import digits.config  # noqa
from digits import utils, log  # noqa
from digits.inference.errors import InferenceError  # noqa
from digits.job import Job  # noqa
from digits.utils.lmdbreader import DbReader  # noqa
#from camvideostream import WebcamVideoStream

# Import digits.config before caffe to set the path
import caffe_pb2  # noqa

logger = logging.getLogger('digits.tools.inference')



def read_labels(labels_file):
    """
    Returns a list of strings

    Arguments:
    labels_file -- path to a .txt file
    """
    if not labels_file:
        print 'WARNING: No labels file provided. Results will be difficult to interpret.'
        return None

    labels = []
    with open(labels_file) as infile:
        for line in infile:
            label = line.strip()
            if label:
                labels.append(label)
    assert len(labels), 'No labels found'
    return labels


def infer(jobs_dir,
          model_id,
          epoch):
    """
    Perform inference on a list of images using the specified model
    """
    # job directory defaults to that defined in DIGITS config
    if jobs_dir == 'none':
        jobs_dir = digits.config.config_value('jobs_dir')

    # load model job
    model_dir = os.path.join(jobs_dir, model_id)
    assert os.path.isdir(model_dir), "Model dir %s does not exist" % model_dir
    model = Job.load(model_dir)

    # load dataset job
    dataset_dir = os.path.join(jobs_dir, model.dataset_id)
    assert os.path.isdir(dataset_dir), "Dataset dir %s does not exist" % dataset_dir
    dataset = Job.load(dataset_dir)
    for task in model.tasks:
        task.dataset = dataset

    # retrieve snapshot file
    task = model.train_task()
    snapshot_filename = None
    epoch = float(epoch)
    if epoch == -1 and len(task.snapshots):
        # use last epoch
        epoch = task.snapshots[-1][1]
        snapshot_filename = task.snapshots[-1][0]
    else:
        for f, e in task.snapshots:
            if e == epoch:
                snapshot_filename = f
                break
    if not snapshot_filename:
        raise InferenceError("Unable to find snapshot for epoch=%s" % repr(epoch))
    return epoch, dataset, model



def inferencia(epoch, dataset, model, image, labels):
    # retrieve image dimensions and resize mode
    gpu = None
    resize = 'store_true'
    layers = 'none'
    input_is_db = 'store_true'
    image_dims = dataset.get_feature_dims()
    height = image_dims[0]
    width = image_dims[1]
    channels = image_dims[2]
    resize_mode = dataset.resize_mode if hasattr(dataset, 'resize_mode') else 'squash'

    n_input_samples = 0  # number of samples we were able to load
#    input_ids = []       # indices of samples within file list
    input_data = []      # sample data

    try:
        if resize:
            image = utils.image.resize_image(
                image,
                height,
                width,
                channels=channels,
                resize_mode=resize_mode)
        else:
            image = utils.image.image_to_array(
                image,
                channels=channels)
#                input_ids.append(idx)
        input_data.append(image)
        n_input_samples = n_input_samples + 1
    except utils.errors.LoadImageError as e:
        print e

    # perform inference
    visualizations = None

    if n_input_samples == 0:
        raise InferenceError("Unable to load any image")
    elif n_input_samples == 1:
        # single image inference
        outputs, visualizations = model.train_task().infer_one(
            input_data[0],
            snapshot_epoch=epoch,
            layers=layers,
            gpu=gpu,
            resize=resize)
        print "one image-----------"
    else:
        if layers != 'none':
            raise InferenceError("Layer visualization is not supported for multiple inference")
        outputs = model.train_task().infer_many(
            input_data,
            snapshot_epoch=epoch,
            gpu=gpu,
            resize=resize)
        print "many images-----------"

    # write to hdf5 file
##    print outputs['softmax']
    for ind in outputs['softmax']:
        a = ind.sum()
#        a = np.sum(ind)
        b = (ind/a)*100
#        print a
#        print (ind/a)*100
#        label = np.array(['center', 'left','right', 'kleft', 'kright'])
        results = {}
        print '{:-^80}'.format(' Prediction for images')
#        print labels, b
        count = 0
        for conf in b:
            print ('%2.4f' % conf), labels[count]
            results[labels[count]] = round(conf / 100.0, 6)
            count+=1
            if count ==7:
                break
        print
#        print results, max(results, key=results.get)

    return max(results, key=results.get), results




def soma_prob(mem_p, gamma = 0.9):
    #gamma Ã© o valor de desconto
    memory_temp_size = len(mem_p.values())

    soma = {}
    for key, value in mem_p.iteritems():
        somador = 0
        for mem_index in range(memory_temp_size):
            somador = (gamma**(memory_temp_size-mem_index-1))*value[mem_index] + somador
#             print mem_p[index][mem_index],
#         print somador
        soma[key] = somador
#     print soma
    return soma





def thread_DNN():
    time.sleep(2)

    while True:
#		script_start_time = time.time()

#		print "FRAME = ", time.time() - script_start_time
        start1 = time.time()
#===============================================================================
        if args2.visionball:
            cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame.all()==None:
            print "No image"
        else:
            type_label, results = inferencia( epoch_r, dataset, model, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), labels)
#            print type_label, results

            for index in labels:
                a = list(mem_p[index])
                del a[0]
                mem_p[index] = a
                b = list(mem_p[index])
                b.append(results[index])
                mem_p[index] = b
        #                print index, mem_p[index]
        #            print type_label, number_label[type_label], results[type_label]
            prob_values = soma_prob(mem_p)
#                print prob_values
            label_max_prob =  max(prob_values, key=prob_values.get)
        #            testlib.escreve_int(1000000, number_label[label_max_prob])
            print label_max_prob
            bkb.write_int(Mem,'DECISION_ACTION_A', number_label[label_max_prob])
            print 'Script took %f seconds.' % (time.time() - start1,)


#===============================================================================
#		print "tempo de varredura = ", time.time() - start

    cap.release()
    cv2.destroyAllWindows()








if __name__ == '__main__':

#    testlib = ctypes.CDLL('./blackboard/blackboard.so') #chama a lybrary que contem as funÃ§Ãµes em c++
#    testlib.using_shared_memory()   #usando a funÃ§Ã£o do c++
#    testlib.leitura_int.restype = ctypes.POINTER(ctypes.c_int) #define o tipo de retorno da funÃ§Ã£o, neste caso a funÃ§Ã£o retorna ponteiro int

#    net_address = "DNN_ballClass"
    net_address = "squeezenet"
    path_jobs_dir = '/home/fei/RoboFEI-DNN/Vision/src/nets'


    """ Instantiate bkb as a shared memory """
    bkb = SharedMemory()

    parser = argparse.ArgumentParser(description='Inference tool - DIGITS')

    # Positional arguments

    parser.add_argument(
        '--db',
        action='store_true',
        help='Input file is a database',
    )

    parser.add_argument(
        '--resize',
        dest='resize',
        action='store_true')

    parser.add_argument(
        '--no-resize',
        dest='resize',
        action='store_false')

    parser.add_argument('--ws', '--ws', action='store_true', help="no servo")
    parser.add_argument('--visionball', '--vb', action="store_true", help = 'Mostra o atual frame da visao')

    parser.set_defaults(resize=True)

    args = vars(parser.parse_args())
    args2 = parser.parse_args()

    #read labels
    labels = read_labels(path_jobs_dir+'/'+net_address+'/labels.txt')
#        print labels
    number_label =  dict(zip(labels, range(len(labels))))
#        print number_label

    #buffer_t = np.zeros((10) , dtype=np.int)
    memory_temp_size = 7
    num_itens = len(labels)
    bucket = [0 for i in range(memory_temp_size)]
    mem_p = {}
    for i in labels:
        mem_p[i] = bucket

    if not args2.ws:
        servo = Servo(583, 330)
#    os.system("v4l2-ctl -d /dev/video0 -c focus_auto=0 && v4l2-ctl -d /dev/video0 -c focus_absolute=0")
    cap = cv2.VideoCapture(0)
    cap.set(3,720) #720 1280 1920
    cap.set(4,480) #480 1024 1080


    try:
        epoch_r, dataset, model = infer(
            jobs_dir = path_jobs_dir,
            model_id = net_address,
            epoch = '-1'
        )
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise


    try:
        thread.start_new_thread(thread_DNN, ())
    except:
        print "Error Thread"

    script_start_time_im=0

    while True:

#        time.sleep(0.05)

#        image_pointer = testlib.leitura_int()
#        image = np.array(image_pointer[2:im_size_array]).reshape( im_height, im_width, 3)
        # Capture frame-by-frame
        ret, image = cap.read()
        frame = image.copy()
#        time.sleep(0.03)
        print 'Read image %f seconds.' % (time.time() - script_start_time_im,)
        script_start_time_im = time.time()
            
#        plt.imshow(image, interpolation='nearest')
#        plt.show()
#        exit()

#        cv2.imwrite('color_img.png', image)
#        exit()

#        cv2.imshow('Image', image/255)
#        if cv2.waitKey(30)>=0:
#            exit()

#        if args2.visionball:
#            cv2.imshow('frame_read',image)

#        print mem_p, results

        # Display the resulting frame
##        cv2.imshow('frame',image)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

