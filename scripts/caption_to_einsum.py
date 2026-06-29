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


def df2einsums(df):

    df = sent2tree(df, labels=["corrected_sentence"])
    einsum_arr = tree2tn(df, ['corrected_sentence_tree'])
    qcs_curried = tn2qc(einsum_arr, ansatz, True)

    einsums = []

    for einsum_arr, param_arr in tqdm(sum(qcs_curried, [])):
        new_param_arr = []
        new_einsum_arr = [[], einsum_arr[1][:]]
        for i in range(len(param_arr)):
            if param_arr[i][1] != '0_dag':
                new_param_arr.append(param_arr[i])
                new_einsum_arr[0].append(einsum_arr[0][i])

        einsum_arr = new_einsum_arr
        new_einsum_arr = [[],""]

        for x in einsum_arr[0]:
            new_einsum_arr[0].append("".join(x))
        new_einsum_arr[0] = str(new_einsum_arr[0])[2:-2].replace(" ","").replace("'","")

        for x in einsum_arr[1]:
            new_einsum_arr[1] += "".join(x)
        # new_einsum_arr[1] = str(new_einsum_arr[1])[2:-2].replace(" ","").replace("'","")

        einsum_arr = f"{new_einsum_arr[0]}->{new_einsum_arr[1]}"
        # einsum_arr = f"{str(einsum_arr[0]).replace("[","").replace("]", "")}->{str(einsum_arr[1]).replace("[","").replace("]", "").replace(" ", "")}"
        # print(einsum_arr)

        
        einsums.append((einsum_arr, new_param_arr))

    print(f"len: {len(einsums)}")
    return einsums

def df2einsums_aro(df):

    df = sent2tree(df, labels=["true_caption"])
    einsum_arr = tree2tn(df, ['true_caption_tree'])
    qcs_curried_pos = tn2qc(einsum_arr, ansatz, True)

    df = sent2tree(df, labels=["false_caption"])
    einsum_arr = tree2tn(df, ['false_caption_tree'])
    qcs_curried_neg = tn2qc(einsum_arr, ansatz, True)

    einsums = []


    for einsum_arr, param_arr in tqdm(sum(qcs_curried_pos, [])):
        einsums.append([(einsum_arr, param_arr)])

    i = 0
    for einsum_arr, param_arr in tqdm(sum(qcs_curried_neg, [])):
        einsums[i].append((einsum_arr, param_arr))
        i +=1



    print(f"len: {len(einsums)}")
    return einsums


df = pd.read_csv(f"{dataset_path}/train.csv")
df = get_valid_images(df, fpath=img_path, img_labels="pos_image_id")
df = get_valid_images(df, fpath=img_path, img_labels="neg_image_id")

einsums = df2einsums(df)
with open(f"{pkl_path}/svo_train_dagless_einsum_as_12.pkl", 'wb') as f:
    pickle.dump(einsums, f)

df = pd.read_csv(f"{dataset_path}/test.csv")
df = get_valid_images(df, fpath=img_path, img_labels="pos_image_id")
df = get_valid_images(df, fpath=img_path, img_labels="neg_image_id")

einsums = df2einsums(df)
with open(f"{pkl_path}/svo_test_dagless_einsum_as_12.pkl", 'wb') as f:
    pickle.dump(einsums, f)

df = pd.read_csv(f"{dataset_path}/val.csv")
df = get_valid_images(df, fpath=img_path, img_labels="pos_image_id")
df = get_valid_images(df, fpath=img_path, img_labels="neg_image_id")

einsums = df2einsums(df)
with open(f"{pkl_path}/svo_valid_dagless_einsum_as_12.pkl", 'wb') as f:
    pickle.dump(einsums, f)

df = pd.read_csv(f"{dataset_path}/svo_probes_swapped.csv")
df = get_valid_images(df, fpath=img_path, img_labels="pos_image_id")
df = get_valid_images(df, fpath=img_path, img_labels="neg_image_id")

einsums = df2einsums(df)
with open(f"{pkl_path}/svo_swap_dagless_einsum_as_12.pkl", 'wb') as f:
    pickle.dump(einsums, f)