import tensorflow as tf
import numpy as np
import pickle
from tqdm.notebook import tqdm 


def make_file_index_list(commit_files):
    
    m_list = []
    for version in commit_files.keys():
        for file_name in commit_files[version]:
            if file_name not in m_list:
                m_list.append(file_name)
    
    file_index = {}
    index = 0
    for f in m_list:
        file_index[f] = index
        index += 1

    file_index_set = {}
    for v in list(commit_files.keys()):
        file_list = []
        for file_name in commit_files[v]:
            file_list.append(file_index[file_name])
        file_index_set[v] = file_list
    
    file_index_list = []
    for version in list(file_index_set.keys()):
        file_index_list.append(file_index_set[version])
    file_index_list
    
    return file_index, file_index_list

def make_version_index(commit_files):
    version_index = {}
    iter_count = 0
    for key in commit_files:
        version_index[key] = iter_count
        iter_count += 1
    return version_index

def make_vector_index_set(vector_set):
    vector_index = 0
    vector_index_set = {}
    i = 0
    for Key in vector_set.keys():
        j = 0
        for _ in vector_set[Key]:
            vector_index_set[vector_index] = [i,j]
            vector_index += 1
            j += 1
        i += 1
    return vector_index_set

def make_Fi_num(commit_files):
    Fi_num = {}
    i = 0
    for key in commit_files.keys():
        Fi_num[i] = len(commit_files[key])
        i += 1
    return Fi_num

def z_init(file_index, file_index_list, all_vector, vector_index_set, Fi_num):
    z_numpy = np.ones([len(file_index), len(all_vector)])
    for z1 in range(z_numpy.shape[0]):
        for z2 in range(z_numpy.shape[1]):
            if z1 not in file_index_list[vector_index_set[z2][0]]:
                z_numpy[z1][z2] = 0
            else:
                z_numpy[z1][z2] = 1 / Fi_num[vector_index_set[z2][0]]
    z_tensor = tf.cast(z_numpy, tf.float32)
    return z_tensor