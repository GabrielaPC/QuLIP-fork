from qiskit import transpile
from qiskit.circuit import Parameter
from collections import defaultdict
import numpy as np
import pickle

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
import pandas as pd
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector, partial_trace


from tn2qiskit_copy import *

from modules.data_processing import *
from modules.model import *
from modules.util import *
from modules.functor import *



ansatz = CustomV5Ansatz(layers=2, obmap = {'n': 1, 'p': 1, 's': 1, 'out': 9})
dataset_path = "C:/Users/Gabriela/Documents/git/QML/dataset/svo_probes"
pkl_path = "C:/Users/Gabriela/Documents/git/QML/svo/curried"

def df2einsums(df):

    df = sent2tree(df, labels=["corrected_sentence"])
    einsum_arr = tree2tn(df, ['corrected_sentence_tree'])
    qcs_curried = tn2qc(einsum_arr, ansatz, True)

    einsums = []

    for einsum_arr, param_arr in tqdm(sum(qcs_curried, [])):
        einsums.append((einsum_arr, param_arr))

    print(f"len: {len(einsums)}")
    return einsums


# df = pd.read_csv(f"{dataset_path}/train.csv")

# einsums = df2einsums(df)
# with open(f"{pkl_path}/svo_train_einsum_as_12.pkl", 'wb') as f:
#     pickle.dump(einsums, f)

# df = pd.read_csv(f"{dataset_path}/test.csv")

# einsums = df2einsums(df)
# with open(f"{pkl_path}/svo_test_einsum_as_12.pkl", 'wb') as f:
#     pickle.dump(einsums, f)

# df = pd.read_csv(f"{dataset_path}/val.csv")
# einsums = df2einsums(df)
# with open(f"{pkl_path}/svo_valid_einsum_as_12.pkl", 'wb') as f:
#     pickle.dump(einsums, f)

df = pd.read_csv(f"{dataset_path}/svo_probes_swapped.csv")
einsums = df2einsums(df)
with open(f"{pkl_path}/svo_swap_einsum_as_12.pkl", 'wb') as f:
    pickle.dump(einsums, f)