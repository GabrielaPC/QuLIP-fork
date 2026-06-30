from collections import defaultdict, Counter

import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from opt_einsum import contract_path
import cotengra as ctg

def sort_tn(tn):
    remaining = list(tn)
    sorted_tn = []

    counts = Counter(sum([tensor[1] for tensor in tn], []))
    boundary_indices = {k for k, v in counts.items() if v == 1}

    for tensor in list(remaining):
        if any(idx in boundary_indices for idx in tensor[1]):
            sorted_tn.append(tensor)
            remaining.remove(tensor)
    
    current_indices = set()
    for tensor in sorted_tn:
        current_indices.update(tensor[1])

    while remaining:
        for tensor in list(remaining):
            if any(idx in current_indices for idx in tensor[1]):
                sorted_tn.append(tensor)
                remaining.remove(tensor)
                current_indices.update(tensor[1])
                break

    return list(reversed(sorted_tn))


def tn_metadata(data_arr):
    max_nq = max_gates = max_width = max_cdepth = 0
    avg_nq = avg_gates = avg_width = avg_cdepth = 0
    N = len(data_arr)
    path_cache = {}
    for einsum_expr, tarr in tqdm(data_arr):
        nq, ngates, cdepth, width = analyse_einsum(einsum_expr, tarr, cache=path_cache)
        max_nq = max(max_nq, nq)
        max_gates = max(max_gates, ngates)
        max_cdepth = max(max_cdepth, cdepth)
        max_width = max(max_width, width)
        avg_nq += nq
        avg_gates += ngates
        avg_cdepth += cdepth
        avg_width += width
    avg_width /= N
    avg_nq /= N
    avg_cdepth /= N
    avg_gates /= N
    return {'max': (max_nq, max_gates, max_cdepth, max_width ), 
            'avg': (int(round(avg_nq)), int(round(avg_gates)), int(round(avg_cdepth)), int(round(avg_width)))}

def analyse_einsum(einsum_expr, tarr, cache={}):
    op_types = tuple(op[1] for op in tarr)
    input_subs, output_sub = einsum_expr
    einsum_str = ','.join([''.join(ten) for ten in input_subs]) + '->' + ','.join([''.join(ten) for ten in output_sub])
    cache_key = (einsum_str, op_types)
    if cache_key in cache:
            return cache[cache_key]

    qubit_depths = defaultdict(int)
    shapes = []
    nq = 0
            
    for i, (subscript, (symbol, op_type)) in enumerate(zip(input_subs, tarr)):
        if symbol is None:
            if op_type == 'sqrt': data_shape = torch.Size([])
            elif op_type == '0':
                data_shape = torch.Size([2])
                nq += 1
            elif op_type == '0_dag':
                data_shape = torch.Size([2])
            elif op_type == 'H': data_shape = torch.Size([2, 2])
            elif op_type == 'CX': data_shape = torch.Size([2, 2, 2, 2])
        else:
            if op_type in ['Rz', 'Rx', 'Ry']: data_shape = torch.Size([2, 2])
            elif op_type in ['CRz', 'CRx', 'CRy']: data_shape = torch.Size([2, 2, 2, 2])
        shapes.append(data_shape)

        if op_type not in ['0', 'sqrt']:
            current_gate_max = 0
            for char in subscript:
                current_gate_max = max(current_gate_max, qubit_depths[char])
            new_depth = current_gate_max + 1
            for char in subscript:
                qubit_depths[char] = new_depth

    cdepth = max(qubit_depths.values()) if qubit_depths else 0
    # opt = ctg.HyperOptimizer(methods=['kahypar', 'greedy'], max_repeats=16, parallel=True)
    # tree = ctg.einsum_tree(einsum_str, *[tuple(int(d) for d in s) for s in shapes], optimize=opt)
    # max_width = tree.contraction_width()
    interleaved_args = []
    for shape, sub in zip(shapes, input_subs):
        interleaved_args.append(shape)
        interleaved_args.append(sub)
    interleaved_args.append(output_sub)
    path_info = contract_path(*interleaved_args, shapes=True)
    max_width = int(np.log2(float(path_info[1].largest_intermediate)))
    ngates = len(tarr) - nq
    cache[cache_key] = (nq, ngates, cdepth, max_width)
    return nq, ngates, cdepth, max_width


def einsum2interleaved(expr):
    if isinstance(expr, str):
        if '->' not in expr:
            raise ValueError("Invalid einsum string format. Missing '->' operator.")
            
        lhs, rhs = expr.split('->')
        input_tensors = [tok.strip() for tok in lhs.split(',') if tok.strip()]
        input_indices = [list(tensor) for tensor in input_tensors]
        out_list = list(rhs.strip())
        
        return (input_indices, out_list)
    
def interleaved2einsum(input_indices, out_list):
    lhs = ",".join("".join(tensor) for tensor in input_indices)
    rhs = "".join(out_list)
    
    return f"{lhs}->{rhs}"

def modal_compose(text_einsum, text_tensor_arr, img_einsum, img_tensor_arr):
    text_indices = text_einsum[0]
    all_text_chars = {char for tensor in text_indices for char in tensor}
    max_text_char = max(all_text_chars) if all_text_chars else 'a'
    char_offset = ord(max_text_char) + 1

    text_outputs = text_einsum[1]
    img_init_positions = [i for i, tensor in enumerate(img_tensor_arr) if tensor == (None, '0')]
    if len(img_init_positions) != len(text_outputs):
        raise ValueError("Dimension Mismatch between Text outputs and Image inputs!")
    
    img_indices = img_einsum[0]
    boundary_wire_map = {img_indices[pos][0]: text_outputs[i] for i, pos in enumerate(img_init_positions)}

    cleaned_img_indices = []
    cleaned_img_tensor_arr = []
    init_set = set(img_init_positions)

    for i, tensor in enumerate(img_indices):
        if i in init_set:
            continue
        remapped_tensor = []
        for wire_char in tensor:
            if wire_char in boundary_wire_map:
                remapped_tensor.append(boundary_wire_map[wire_char])
            else:
                new_char = "".join(chr(ord(c) + char_offset) for c in wire_char)
                remapped_tensor.append(new_char)
        cleaned_img_indices.append(remapped_tensor)
        cleaned_img_tensor_arr.append(img_tensor_arr[i])
    unified_indices = text_indices + cleaned_img_indices
    unified_tensor_arr = text_tensor_arr + cleaned_img_tensor_arr

    final_outputs = []
    for wire_char in img_indices[-1]:
        if wire_char in boundary_wire_map:
            final_outputs.append(boundary_wire_map[wire_char])
        else:
            new_char = "".join(chr(ord(c) + char_offset) for c in wire_char)
            final_outputs.append(new_char)
    return (unified_indices, final_outputs), unified_tensor_arr