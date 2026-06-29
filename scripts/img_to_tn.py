from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import pandas as pid
import numpy as np
import torch, os, pickle, mlflow, time, sys
from math import acos


from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService
# from qiskit_ibm_runtime.fake_provider import FakeMiami

from qiskit_aer import AerSimulator
from qiskit import transpile


root_path = os.getcwd()
os.chdir(root_path)
sys.path.insert(0, root_path)


from tn2qiskit_copy import *

from modules.data_processing import *
from modules.model import *
from modules.util import *
from modules.functor import *


root_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'QML')

img_path = os.path.join(root_path, 'img_encodings/SVO')

dataset_path = os.path.join(root_path, 'dataset/svo_probes')
swap_img_path = os.path.join(root_path, 'dataset/svo_probes_images')

# print("Loading test:")
# df = pd.read_csv(f"{dataset_path}/test.csv")
# df = get_valid_images(df, fpath=swap_img_path, img_labels="pos_image_id")
# df = get_valid_images(df, fpath=swap_img_path, img_labels="neg_image_id")

# test_pos_img = load_embeddings_from_df(df, fpath=swap_img_path, img_labels="pos_image_id")
# test_neg_img = load_embeddings_from_df(df, fpath=swap_img_path, img_labels="neg_image_id")

# store_pkl(test_pos_img, os.path.join(img_path, 'svo_imgenc_test_pos_512.pkl'))
# store_pkl(test_neg_img, os.path.join(img_path, 'svo_imgenc_test_neg_512.pkl'))
# print("Saved embedings")

# test_pos_img = tuple(test_pos_img)
# test_neg_img = tuple(test_neg_img)

print("Loading train:")
df = pd.read_csv(f"{dataset_path}/train.csv")
df = get_valid_images(df, fpath=swap_img_path, img_labels="pos_image_id")
df = get_valid_images(df, fpath=swap_img_path, img_labels="neg_image_id")

train_pos_img = load_embeddings_from_df(df, fpath=swap_img_path, img_labels="pos_image_id")

train_neg_img = load_embeddings_from_df(df, fpath=swap_img_path, img_labels="neg_image_id")
# del swap_img

store_pkl(train_pos_img, os.path.join(img_path, 'svo_imgenc_train_pos_512.pkl'))
store_pkl(train_neg_img, os.path.join(img_path, 'svo_imgenc_train_neg_512.pkl'))
print("Saved embedings")

train_pos_img = tuple(train_pos_img)
train_neg_img = tuple(train_neg_img)

print("Loading valid:")
df = pd.read_csv(f"{dataset_path}/val.csv")
df = get_valid_images(df, fpath=swap_img_path, img_labels="pos_image_id")
df = get_valid_images(df, fpath=swap_img_path, img_labels="neg_image_id")

valid_pos_img = load_embeddings_from_df(df, fpath=swap_img_path, img_labels="pos_image_id")
valid_neg_img = load_embeddings_from_df(df, fpath=swap_img_path, img_labels="neg_image_id")

store_pkl(valid_pos_img, os.path.join(img_path, 'svo_imgenc_valid_pos_512.pkl'))
store_pkl(valid_neg_img, os.path.join(img_path, 'svo_imgenc_valid_neg_512.pkl'))
print("Saved embedings")

valid_pos_img = tuple(valid_pos_img)
valid_neg_img = tuple(valid_neg_img)

# train_pos_img = load_pkl(os.path.join(img_path, 'svo_imgenc_train_pos_512.pkl'))
# train_neg_img = load_pkl(os.path.join(img_path, 'svo_imgenc_train_neg_512.pkl'))
# train_pos_img = tuple(train_pos_img)
# train_neg_img = tuple(train_neg_img)



img_dataset = train_pos_img
img_dataset += train_neg_img
img_dataset = [x.float().flatten() for x in img_dataset]

img_ansatz = ImageFeatureMap(img_dataset, k=9)
del img_dataset

print("Beginning Ansatz")
# train_pos_img_tn = [img_ansatz.ansatz(img.flatten().float()) for img in train_pos_img]
# store_pkl(train_pos_img_tn,os.path.join(img_path, 'svo_train_pos_tns.pkl'))
# train_neg_img_tn = [img_ansatz.ansatz(img.flatten().float()) for img in train_neg_img]
# store_pkl(train_neg_img_tn,os.path.join(img_path, 'svo_train_neg_tns.pkl'))
# print("Train done")

# test_pos_img_tn = [img_ansatz.ansatz(img.flatten().float()) for img in test_pos_img]
# store_pkl(test_pos_img_tn,os.path.join(img_path, 'svo_test_pos_tns.pkl'))
# test_neg_img_tn = [img_ansatz.ansatz(img.flatten().float()) for img in test_neg_img]
# store_pkl(test_neg_img_tn,os.path.join(img_path, 'svo_test_neg_tns.pkl'))
# print("Test done")

valid_pos_img_tn = [img_ansatz.ansatz(img.flatten().float()) for img in valid_pos_img]
store_pkl(valid_pos_img_tn,os.path.join(img_path, 'svo_valid_pos_tns.pkl'))
valid_neg_img_tn = [img_ansatz.ansatz(img.flatten().float()) for img in valid_neg_img]
store_pkl(valid_neg_img_tn,os.path.join(img_path, 'svo_valid_neg_tns.pkl'))
print("Valid done")

