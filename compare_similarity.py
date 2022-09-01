import h5py
from scipy import spatial
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import spatial
from tqdm import tqdm
import pandas as pd

from numpy import dot
from numpy.linalg import norm
import os

f = "CONFORMER_clevr_change.hdf5"
f_t = "CONFORMER_clevr_change_t.hdf5"

f_pca = "CONFORMER_clevr_change_afterpca.hdf5"
f_t_pca = "CONFORMER_clevr_change_t_afterpca.hdf5"

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def get_similarity(hdf5_file = f_t):

    non_semantic_index = 40000
    semantic_index = 80000
    print("filename: ", hdf5_file)
    f1 = h5py.File(hdf5_file, 'r')
    print("read done")
    key_list = list(f1.keys())
    print("key_list done")

    nonsemantic_sim = 0
    semantic_sim = 0
    testindex = 0
    saved_result = {'keyname': [], 'nonsemantic': [], 'semantic': []}
    for default_key, non_semantic_key, semantic_key in tqdm(zip(key_list[:non_semantic_index],
                                                                key_list[non_semantic_index:semantic_index],
                                                                key_list[semantic_index:]), total=40000):
        testindex += 1
        # print("default_keyname: ", default_key)
        # print("non_semantic_keyname: ", non_semantic_key)
        # print("semantic_keyname: ", semantic_key)
        default = f1[default_key][()].flatten()
        non_semantic = f1[non_semantic_key][()].flatten()
        semantic = f1[semantic_key][()].flatten()
        val1 = 1 - spatial.distance.cosine(default, non_semantic)
        val2 = 1 - spatial.distance.cosine(default, semantic)
        saved_result['keyname'].append(default_key)
        saved_result['nonsemantic'].append(val1)
        saved_result['semantic'].append(val2)
        # print(default)
        # print(non_semantic)
        # print(semantic)
        # if testindex > 10:
        #     break
        # print("default vs nonsemantic: ", val1)
        # print("default vs semantic: ", val2)
        # print("------------------------------------------------------------------------------------------------------")

    df = pd.DataFrame(saved_result)
    filename = hdf5_file.split(".")[0]
    df.to_excel(filename +"_similarity.xlsx")
    print("file save done")

# f1 = h5py.File(f, 'r')
# print("read done")
# key_list = list(f1.keys())
# print("key_list done")
#
# for index in range(10):
#     key = key_list[index]
#     key2 = key_list[index + 40000]
#     default =  f1[key][()].flatten()
#     nonsemantic = f1[key2][()].flatten()
#     print(1 - spatial.distance.cosine(default, nonsemantic))





get_similarity(hdf5_file=f_t)
get_similarity(hdf5_file=f)