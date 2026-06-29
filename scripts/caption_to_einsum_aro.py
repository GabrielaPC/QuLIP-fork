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

# root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
root_path = os.getcwd()
sys.path.insert(0, root_path)

os.chdir(root_path)

print(os.getcwd())

# from scripts.tn2qiskit_copy import *

from modules.data_processing import *
from modules.model import *
from modules.util import *
from modules.functor import *


ansatz = CustomV5Ansatz(layers=2, obmap = {'n': 1, 'p': 1, 's': 1, 'out': 9})

root_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'QML')


relation = False
# relation = True

if relation:
    dataset_path = os.path.join(root_path, 'dataset/aro_datasets/visual_genome_relation')
    pkl_path = os.path.join(root_path, 'aro/curried/relation')
else:
    dataset_path = os.path.join(root_path, 'dataset/aro_datasets/visual_genome_attribution')
    pkl_path = os.path.join(root_path, 'aro/curried/attribution')
img_path = os.path.join(root_path,"dataset/aro_images")




def df2einsums_aro(df):

    df = sent2tree(df, labels=["true_caption"])
    einsum_arr = tree2tn(df, ['true_caption_tree'])
    qcs_curried_pos = tn2qc(einsum_arr, ansatz, True)

    df = sent2tree(df, labels=["false_caption"])
    einsum_arr = tree2tn(df, ['false_caption_tree'])
    qcs_curried_neg = tn2qc(einsum_arr, ansatz, True)

    einsums = []


    for einsum_arr, param_arr in tqdm(sum(qcs_curried_pos, [])):
        for x in einsum_arr[0]:
            new_einsum_arr[0].append("".join(x))
        new_einsum_arr[0] = str(new_einsum_arr[0])[2:-2].replace(" ","").replace("'","")

        for x in einsum_arr[1]:
            new_einsum_arr[1] += "".join(x)

        einsum_arr = f"{new_einsum_arr[0]}->{new_einsum_arr[1]}"

        einsums.append([(einsum_arr, param_arr)])


    j = 0
    for einsum_arr, param_arr in tqdm(sum(qcs_curried_neg, [])):
        new_einsum_arr = [[],""]

        for x in einsum_arr[0]:
            new_einsum_arr[0].append("".join(x))
        new_einsum_arr[0] = str(new_einsum_arr[0])[2:-2].replace(" ","").replace("'","")

        for x in einsum_arr[1]:
            new_einsum_arr[1] += "".join(x)

        einsum_arr = f"{new_einsum_arr[0]}->{new_einsum_arr[1]}"

        einsums[j].append((einsum_arr, param_arr))
        j +=1

    print(f"len: {len(einsums)}")
    return einsums


# df = pd.read_csv(f"{dataset_path}/train.csv")
df = pd.read_json(f"{dataset_path}/train.json")
df = get_valid_images(df, fpath=img_path, img_labels="image_id")

einsums = df2einsums_aro(df)
with open(f"{pkl_path}/aro_train_einsum_as_12.pkl", 'wb') as f:
    pickle.dump(einsums, f)

# df = pd.read_csv(f"{dataset_path}/test.csv")
df = pd.read_json(f"{dataset_path}/test.json")
df = get_valid_images(df, fpath=img_path, img_labels="image_id")

einsums = df2einsums_aro(df)
with open(f"{pkl_path}/aro_test_einsum_as_12.pkl", 'wb') as f:
    pickle.dump(einsums, f)

# df = pd.read_csv(f"{dataset_path}/val.csv")
df = pd.read_json(f"{dataset_path}/val.json")
df = get_valid_images(df, fpath=img_path, img_labels="image_id")

einsums = df2einsums_aro(df)
with open(f"{pkl_path}/aro_valid_einsum_as_12.pkl", 'wb') as f:
    pickle.dump(einsums, f)
