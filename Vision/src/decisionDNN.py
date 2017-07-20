#coding: utf-8
#!/usr/bin/env python2



"""
Classify an image using a model archive file
"""


import argparse
import os
import tarfile
import tempfile
import time
import zipfile
from math import log,exp,tan,radians
import thread
from defs import *
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



def classify(image_files, net, transformer,
             mean_file=None, labels=None, batch_size=None):
    """
    Classify some images against a Caffe model and print the results

    Arguments:
    caffemodel -- path to a .caffemodel
    deploy_file -- path to a .prototxt
    image_files -- list of paths to images

    Keyword arguments:
    mean_file -- path to a .binaryproto
    labels_file path to a .txt file
    use_gpu -- if True, run inference on the GPU
    """
#    if channels == 3:
#        mode = 'RGB'
#    elif channels == 1:
#        mode = 'L'
#    else:
#        raise ValueError('Invalid number for channels: %s' % channels)
    images = [image_files]

    # Classify the image
    scores = forward_pass(images, net, transformer, batch_size=batch_size)

    #
    # Process the results
    #
    results = {}

    indices = (-scores).argsort()  # take top 9 results
    classifications = []
    for image_index, index_list in enumerate(indices):
        result = []
        for i in index_list:
            # 'i' is a category in labels and also an index into scores
            if labels is None:
                label = 'Class #%s' % i
            else:
                label = labels[i]
            result.append((label, round(100.0 * scores[image_index, i], 4)))
        classifications.append(result)

    for index, classification in enumerate(classifications):
        print '{:-^80}'.format(' Prediction for image ')
        for label, confidence in classification:
            print '{:9.4%} - "{}"'.format(confidence / 100.0, label)
            results[label] = confidence / 100.0
        print
    return classification[0][0], results

def unzip_archive(archive):
    """
    Unzips an archive into a temporary directory
    Returns a link to that directory

    Arguments:
    archive -- the path to an archive file
    """
    assert os.path.exists(archive), 'File not found - %s' % archive

    tmpdir = os.path.join(tempfile.gettempdir(), os.path.basename(archive))
    assert tmpdir != archive  # That wouldn't work out

    if os.path.exists(tmpdir):
        # files are already extracted
        pass
    else:
        if tarfile.is_tarfile(archive):
            print 'Extracting tarfile ...'
            with tarfile.open(archive) as tf:
                tf.extractall(path=tmpdir)
        elif zipfile.is_zipfile(archive):
            print 'Extracting zipfile ...'
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(path=tmpdir)
        else:
            raise ValueError('Unknown file type for %s' % os.path.basename(archive))
    return tmpdir

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
    time.sleep(1)

    while True:
#		script_start_time = time.time()

#		print "FRAME = ", time.time() - script_start_time
        start1 = time.time()
#===============================================================================
        if args2.visionball:
            cv2.imshow('frame',frame)


        if frame==None:
            print "No image"
        else:
            type_label, results = classify(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), net, transformer,
                                mean_file=mean_file, labels=labels,
                                batch_size=None)

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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()








if __name__ == '__main__':

#    testlib = ctypes.CDLL('./blackboard/blackboard.so') #chama a lybrary que contem as funÃ§Ãµes em c++
#    testlib.using_shared_memory()   #usando a funÃ§Ã£o do c++
#    testlib.leitura_int.restype = ctypes.POINTER(ctypes.c_int) #define o tipo de retorno da funÃ§Ã£o, neste caso a funÃ§Ã£o retorna ponteiro int

    
    """ Instantiate bkb as a shared memory """
    bkb = SharedMemory()

    parser = argparse.ArgumentParser(description='Classification example using an archive - DIGITS')

    # Positional arguments
    parser.add_argument('archive', help='Path to a DIGITS model archive')
    #parser.add_argument('image_file', nargs='+', help='Path[s] to an image')
    # Optional arguments
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--nogpu', action='store_true', help="Don't use the GPU")

    parser.add_argument('--ws', '--ws', action='store_true', help="no servo")
    parser.add_argument('--visionball', '--vb', action="store_true", help = 'Mostra o atual frame da visao')

    args = vars(parser.parse_args())
    args2 = parser.parse_args()

    tmpdir = unzip_archive(args['archive'])
    caffemodel = None
    deploy_file = None
    mean_file = None
    labels_file = None
    for filename in os.listdir(tmpdir):
        full_path = os.path.join(tmpdir, filename)
        if filename.endswith('.caffemodel'):
            caffemodel = full_path
        elif filename == 'deploy.prototxt':
            deploy_file = full_path
        elif filename.endswith('.binaryproto'):
            mean_file = full_path
        elif filename == 'labels.txt':
            labels_file = full_path
        else:
            print 'Unknown file:', filename

    assert caffemodel is not None, 'Caffe model file not found'
    assert deploy_file is not None, 'Deploy file not found'

    # Load the model and images
    net = get_net(caffemodel, deploy_file, use_gpu=False)
    transformer = get_transformer(deploy_file, mean_file)
    _, channels, height, width = transformer.inputs['data']
    labels = read_labels(labels_file)

#    image_pointer = testlib.leitura_int()
#    im_width = image_pointer[0]
#    im_height = image_pointer[1]
#    im_size_array = im_width*im_height*3+2
#    image = np.zeros([im_width,im_height,3])

    #create index from label to use in decicion action
    number_label =  dict(zip(labels, range(len(labels))))
    print number_label

    #buffer_t = np.zeros((10) , dtype=np.int)
    memory_temp_size = 7
    num_itens = len(labels)
    bucket = [0 for i in range(memory_temp_size)]
    mem_p = {}
    for i in labels:
        mem_p[i] = bucket

    if not args2.ws:
        servo = Servo(436, 700)
#    os.system("v4l2-ctl -d /dev/video0 -c focus_auto=0 && v4l2-ctl -d /dev/video0 -c focus_absolute=0")
    cap = cv2.VideoCapture(0)
    cap.set(3,720) #720 1280 1920
    cap.set(4,480) #480 1024 1080


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
#        time.sleep(0.05)
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

        if args2.visionball:
            cv2.imshow('frame_read',image)

#        print mem_p, results

        # Display the resulting frame
##        cv2.imshow('frame',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

