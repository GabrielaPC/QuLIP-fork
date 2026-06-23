from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import pandas as pid
import numpy as np
import torch, os, clip, pickle, mlflow, time, sys
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
# os.chdir(os.path.join(os.getcwd(), 'git/QuLIP'))

# print(os.getcwd())

from modules.data_processing import *
from modules.model import *
from modules.quantum import *

def run_circuit(qc, param_dict, num_post_select):
    qc = transpile(qc,backend)
    job = sampler.run([(qc, param_dict)], shots=shots)

    result = job.result()
    counts = result[0].join_data().get_counts()

    post_selected_shots = counts.get("0"*(num_post_select+1),0)

    return acos(post_selected_shots/(post_selected_shots + counts.get(f'1{"0"*(num_post_select)}',0)))



NQUBITS = 5
NLAYERS = 3
IMG_DIM = 512

backend = AerSimulator()
sampler = Sampler(backend)

usable_shots = 1000
shots = 64 * usable_shots


root_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'QML')
txt_path = os.path.join(root_path, 'svo/curried')
img_path = os.path.join(root_path, 'img_encodings/SVO/')


test_einsum = load_pkl(os.path.join(txt_path, 'svo_swap_einsum_as_12.pkl'))

IMG_DIM = 512

train_pos_img = load_pkl(os.path.join(img_path, 'svo_imgenc_train_pos_512.pkl'))
train_neg_img = load_pkl(os.path.join(img_path, 'svo_imgenc_train_neg_512.pkl'))

test_pos_img = load_pkl(os.path.join(img_path, 'svo_test_pos_tns.pkl'))
test_neg_img = load_pkl(os.path.join(img_path, 'svo_test_neg_tns.pkl'))


img_dataset = train_pos_img + train_neg_img
img_dataset = [x.float().flatten() for x in img_dataset]
img_ansatz = ImageFeatureMap(img_dataset, k=9)


test_einsum = tuple(test_einsum)
test_pos_img = tuple(test_pos_img)
test_neg_img = tuple(test_neg_img)

params = load_pkl(os.path.join(root_path, 'svo_runs/06_19_17;11;51/model.lt'))
all_params_dict = {}
img_params_dict = {}
for i in range(len(params['model_symbols'])):
    symbol = params['model_symbols'][i]
    if type(symbol) is str:
        if "img_" in symbol:
            img_params_dict[params['model_symbols'][i]] = float(params['model_weights'][i].detach())
        else:
            all_params_dict[params['model_symbols'][i]] = float(params['model_weights'][i].detach())


data_size = len(test_einsum)
acc = 0
for i in range(data_size):
    qc_txt, output_qubits, params_dict = tn2qiskit(test_einsum[i][0], test_einsum[i][1], meas_output=False, all_params_dict=all_params_dict)

    for i in range(qc_txt.num_qubits):
        if i not in output_qubits:
          qc_txt.measure(i, i)  
    

    qc_pos_img, img_output, _ = tn2qiskit(expr2list(test_pos_img[i][0]), test_pos_img[i][1], meas_output=False)
    qc_neg_img, img_output,_ = tn2qiskit(expr2list(test_neg_img[i][0]), test_neg_img[i][1], meas_output=False)

    params_dict.update(img_params_dict)

    qc_pos = qc_txt.copy()
    qreg_txt = qc_pos.qregs[0]
    qreg_img = qc_pos_img.qregs[0]

    qreg_anc = QuantumRegister(1, "q_anc")
    creg_anc = ClassicalRegister(1, "c_anc")

    qc_pos.add_register(qreg_img, qreg_anc, creg_anc)
    qc_pos.compose(qc_pos_img, qreg_img, inplace=True)
    qc_pos.h(qreg_anc)
    for i in range(len(output_qubits)):
        qc_pos.cswap(qreg_anc, qreg_txt[output_qubits[i]], qreg_img[i])
    qc_pos.h(qreg_anc)
    qc_pos.measure(qreg_anc, creg_anc)

    # qc_pos.draw("mpl")
    pos_f = run_circuit(qc_pos, params_dict, qc_txt.num_qubits)

    qc_neg = qc_txt.copy()
    qreg_txt = qc_neg.qregs[0]
    qreg_img = qc_neg_img.qregs[0]

    qreg_anc = QuantumRegister(1, "q_anc")
    creg_anc = ClassicalRegister(1, "c_anc")

    qc_neg.add_register(qreg_img, qreg_anc, creg_anc)
    qc_neg.compose(qc_neg_img, qreg_img, inplace=True)
    qc_neg.h(qreg_anc)
    for i in range(len(output_qubits)):
        qc_neg.cswap(qreg_anc, qreg_txt[output_qubits[i]], qreg_img[i])
    qc_neg.h(qreg_anc)
    qc_neg.measure(qreg_anc, creg_anc)

    # qc_pos.draw("mpl")
    neg_f = run_circuit(qc_neg, params_dict, qc_txt.num_qubits)

    if pos_f > neg_f:
        acc +=1

print(f"Final accuracy:{acc/data_size}")
