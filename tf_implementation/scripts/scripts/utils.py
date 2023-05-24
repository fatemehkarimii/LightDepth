from glob import glob
import os 
import tensorflow as tf
import numpy as np

def good(image):
    num_channels = image.shape[-1]
    means = [123.68, 116.779, 103.939]
    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] += means[i]
    temp = tf.concat(axis=2, values=channels)
    return (temp - np.amin(temp))/np.amax(temp)

def get_ground_truth_path(t):
    u = t.split('/')
    u[1] = 'data_depth_annotated'
    u.pop(2)
    u.insert(3,'proj_depth/groundtruth')
    u.pop(-2)
    return '/'.join(u)

def update_train_filenames_file(config):
    if os.path.exists(config.train_filenames_file):
        basename = os.path.basename(config.train_filenames_file)
        dirname = os.path.dirname(config.train_filenames_file)
        oldfile_name = os.path.join(dirname,'deprecated_'+basename)
        os.rename(config.train_filenames_file,oldfile_name)
    print('reading the test filenames file')
    test_data = list(open(config.test_filenames_file,'r'))
    print('splitting the test filenames file')
    test_data = set([t.split(' ')[0] for t in test_data])
    print('reading training data recursively')
    train_data = glob(os.path.join(config.input_data_path,
                                   '*/*[!()]/image_02/data/*.png'),
                      recursive=True)
    intersection_with_test_data = test_data.intersection(train_data)
    
    print(f'len test {len(test_data)}')
    print(f'len train {len(train_data)-len(intersection_with_test_data)}')
    print(f'len inter {len(intersection_with_test_data)}')
    for item in intersection_with_test_data:
        train_data.remove(item)
    gt_data = [get_ground_truth_path(p) for p in train_data]
    missing_counter = 0
    for v in gt_data:
        if not os.path.exists(v):
            missing_counter+=1
    print(f'number of missing examples {missing_counter}')
    print('generating paths')
    train_data = [u+' '+v for u,v in zip(train_data,gt_data) if os.path.exists(v)]
    print(f'writing in {config.train_filenames_file}')
    with open(config.train_filenames_file,'w') as buffer:
        buffer.write('\n'.join(train_data))




