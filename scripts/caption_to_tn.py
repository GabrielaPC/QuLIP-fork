from qiskit import transpile
from qiskit.circuit import Parameter
from collections import defaultdict
import numpy as np
import pickle, os, sys

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
import pandas as pd
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector, partial_trace

root_path = os.getcwd()

# root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0, root_path)

os.chdir(root_path)

print(os.getcwd())

# from scripts.tn2qiskit_copy import *

from modules.data_processing import *
from modules.model import *
from modules.util import *
from modules.functor import *

root_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'QML')

ansatz = CustomV5Ansatz(layers=2, obmap = {'n': 1, 'p': 1, 's': 1, 'out': 9})
dataset_path = os.path.join(root_path, "dataset/svo_probes")
pkl_path = os.path.join(root_path,"svo/curried")
img_path = os.path.join(root_path,"dataset/svo_probes_images")


def df2tn(df):
    df = sent2tree(df, labels=["corrected_sentence"])
    einsum_arr = tree2tn(df, ['corrected_sentence_tree'])
    return df

df = pd.read_csv(f"{dataset_path}/train.csv")
df = get_valid_images(df, fpath=img_path, img_labels="pos_image_id")
df = get_valid_images(df, fpath=img_path, img_labels="neg_image_id")

df = df2tn(df)
df.to_csv(f"{dataset_path}/train.csv")


df = pd.read_csv(f"{dataset_path}/test.csv")
df = get_valid_images(df, fpath=img_path, img_labels="pos_image_id")
df = get_valid_images(df, fpath=img_path, img_labels="neg_image_id")

df = df2tn(df)
df.to_csv(f"{dataset_path}/test.csv")

df = pd.read_csv(f"{dataset_path}/val.csv")
df = get_valid_images(df, fpath=img_path, img_labels="pos_image_id")
df = get_valid_images(df, fpath=img_path, img_labels="neg_image_id")

df = df2tn(df)
df.to_csv(f"{dataset_path}/val.csv")

df = pd.read_csv(f"{dataset_path}/svo_probes_swapped.csv")
df = get_valid_images(df, fpath=img_path, img_labels="pos_image_id")
df = get_valid_images(df, fpath=img_path, img_labels="neg_image_id")

df = df2tn(df)
df.to_csv(f"{dataset_path}/svo_probes_swapped.csv")