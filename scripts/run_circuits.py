from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import pandas as pid
import numpy as np
import torch, os, clip, pickle, mlflow, time, sys, argparse, yaml
from math import acos


from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeMiami


root_path = os.getcwd()
sys.path.insert(0, root_path)

os.chdir(root_path)

print(os.getcwd())
# uv run python train.py --config configs/tensor_network.yaml

from modules.data_processing import *
from modules.quantum import *


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to experiment config YAML')
args = parser.parse_args()

with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

IMG_QUBITS = config.get("img_dim", 512)

if config['backend'] == 'IBM':
    backend = FakeMiami()
backend = AerSimulator()

backend.set_options(
    max_parallel_threads = 0, 
    max_parallel_experiments = 0,
)
sampler = Sampler(backend)

usable_shots = config.get("usable_shots", 1000)
shots = 64 * usable_shots

test_einsum = load_pkl(config['txt_path'])

test_pos_img = load_pkl(config['img_path']['pos_path'])
test_neg_img = load_pkl(config['img_path']['neg_path'])

test_einsum = tuple(test_einsum)
test_pos_img = tuple(test_pos_img)
test_neg_img = tuple(test_neg_img)

params = load_pkl(config['model_path'])
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
print(f"data_size:{data_size}")
acc = 0

ansatz = config.get("img_ansatz", False)

pos_circs = []
neg_circs = []
# for i in range(data_size):
for i in range(3):
    try:
        qc_txt, output_qubits, params_dict = tn2qiskit(expr2list(test_einsum[i][0]), test_einsum[i][1], meas_output=False, all_params_dict=all_params_dict)

        for i in range(qc_txt.num_qubits):
            if i not in output_qubits:
                qc_txt.measure(i,i)
        if ansatz:
            qc_pos_img, img_output, _ = tn2qiskit(expr2list(test_pos_img[i][0]), test_pos_img[i][1], meas_output=False)
            qc_neg_img, img_output,_ = tn2qiskit(expr2list(test_neg_img[i][0]), test_neg_img[i][1], meas_output=False)
    
            params_dict.update(img_params_dict)
        else:
            qc_pos_img = QuantumCircuit(IMG_QUBITS, 0)
            test_pos_img[i] = np.array(test_pos_img[i]/np.linalg.norm(test_pos_img[i]))
            qc_pos_img.initialize(test_pos_img[i])
            
            qc_neg_img = QuantumCircuit(IMG_QUBITS, 0)
            test_neg_img[i] = np.array(test_neg_img[i]/np.linalg.norm(test_neg_img[i]))
            qc_neg_img.initialize(test_neg_img[i])
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
        # pos_f = run_circuit(qc_pos, params_dict, qc_txt.num_qubits)

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

        # qc_pos.draw("mpl", fold=-1)
        # neg_f = run_circuit(qc_neg, params_dict, qc_txt.num_qubits)


        # Appending last to guarantee that both circuits where created without errors.
        qc_pos = transpile(qc_pos,backend)
        qc_neg = transpile(qc_neg,backend)

        pos_circs.append((qc_pos, params_dict))
        neg_circs.append((qc_neg, params_dict))

    except Exception as e:
        print(f"circuit {i} falied: {e}")
        data_size -= 1

print("All circuits created, running...")
pos_f = run_circuits(pos_circs,sampler, shots)
neg_f = run_circuits(neg_circs,sampler, shots)

cum_acc = acc_circ(pos_f, neg_f)
print(f"Final accuracy:{cum_acc}")
