import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
from ase.io import read
import os
import sys
import json
from mace.calculators import MACECalculator
import argparse
import numpy as np
from pymatgen.core import Structure
"""Use:
    python violin_compare.py \
        --traj1 <traj_dir_1> --model1 <model_1.model> --label1 N-50pts \
        --traj2 <traj_dir_2> --model2 <model_2.model> --label2 FT5 \
        [--output figure.svg]
"""

Z_to_element = { # GPT generated list
    1: 'H',   2: 'He',
    3: 'Li',  4: 'Be',  5: 'B',   6: 'C',   7: 'N',   8: 'O',   9: 'F',  10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',  16: 'S',  17: 'Cl', 18: 'Ar',
    19: 'K',  20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V',  24: 'Cr', 25: 'Mn', 26: 'Fe',
    27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se',
    35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y',  40: 'Zr', 41: 'Nb', 42: 'Mo',
    43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I',  54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce',
    59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy',
    67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W',
    75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb',
    83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
    91: 'Pa', 92: 'U',  93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf',
    99: 'Es',100: 'Fm',101: 'Md',102: 'No',103: 'Lr',104: 'Rf',105: 'Db',106: 'Sg',
   107: 'Bh',108: 'Hs',109: 'Mt',110: 'Ds',111: 'Rg',112: 'Cn',113: 'Nh',114: 'Fl',
   115: 'Mc',116: 'Lv',117: 'Ts',118: 'Og'
}
SMALL_N_ELEMENTS = {} 
# Fig style
COLORS = ['#E9C5DD', '#3B3657']
plt.style.use('default')
plt.rc('font', size=18)
plt.rc('axes', titlesize=22)
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=15)
plt.rcParams['axes.linewidth'] = 3
def get_respective_files(root_dir):
    """Takes in directory, returns all pairs of POSCAR/vasprun.
    Params:
        Root directory
    Returns:
        matrix of form: [[.../POSCAR, .../vasprun.xml], ...]
    """
    poscars = {}
    vaspruns = {}
    for dirpath, _, filenames in os.walk(root_dir, followlinks=True):
        for fn in filenames:
            parent = os.path.basename(dirpath)
            if fn == "POSCAR":
                poscars[parent] = os.path.join(dirpath, fn)
            elif fn == "vasp_info.json":
                vaspruns[parent] = os.path.join(dirpath, fn)
    return [(poscars[k], vaspruns[k]) for k in poscars if k in vaspruns]
def get_dft_forces(json_path):
    """
    Takes in a JSON file from write_info_to_file / write_vasprun_json,
    returns (per_frame, success_flag, stress)
    
    per_frame: Nx6 array [[x,y,z,fx,fy,fz], ...]
    """
    with open(json_path) as f:
        data = json.load(f)
    try:
        structure = Structure.from_dict(data["structure"])
        forces = np.array(data["forces"])
        atomic_numbers = np.array([site.specie.Z for site in structure.sites])
        return forces, atomic_numbers, True
    except Exception as e:
        print(f"  ! Error reading {json_path}: {e}")
        return None, None, False
def compute_mace_forces(poscar_path, calc):
    atoms = read(poscar_path)
    atoms.calc = calc
    return atoms.get_forces()


def evaluate_self(traj_dir, model_path, label, device):
    """
    Self-evaluation: run model on every POSCAR in traj_dir, compare to DFT.
    Returns dict mapping atomic number -> 1D array of per-atom force-magnitude errors.
    """
    pairs = get_respective_files(traj_dir)
    calc = MACECalculator(model_path, device=device)
    all_errors = []
    all_Z = []
    n_total = len(pairs)
    for i, (poscar, vasp_json) in enumerate(pairs, start=1):
        dft_forces, Z, _ = get_dft_forces(vasp_json)
        mace_forces = compute_mace_forces(poscar, calc)
        err_components = np.abs(dft_forces - mace_forces)
        all_errors.append(err_components.flatten())
        all_Z.append(np.repeat(Z, 3))
    all_errors = np.concatenate(all_errors)
    all_Z = np.concatenate(all_Z)
    per_element = {}
    for z in np.unique(all_Z):
        per_element[int(z)] = all_errors[all_Z == z]
    return per_element
def plot_violins(errors_by_model, output_path, log_scale=False):
    elements_to_show = {z for per_elem in errors_by_model.values() for z in per_elem}
    n_elements = len(elements_to_show)
    n_models = len(errors_by_model)
    model_names = list(errors_by_model.keys())
    model_colors = {name: COLORS[i % len(COLORS)] for i, name in enumerate(model_names)}
    fig, ax = plt.subplots(figsize=(max(7, n_elements * 1.4), 5))
    width_per_violin = 0.8/n_models
    positions = []
    data_for_plot = []
    fill_colors = []
    for i, z in enumerate(elements_to_show):
        for j, model_name in enumerate(model_names):
            errs = errors_by_model[model_name].get(z, np.array([]))
            if len(errs) == 0:
                continue
            offset = (j - (n_models - 1) / 2) * width_per_violin
            positions.append(i + offset)
            data_for_plot.append(np.log10(errs + 1e-6) if log_scale else errs)
            fill_colors.append(model_colors[model_name])
    parts = ax.violinplot(data_for_plot, positions=positions,
                          widths=width_per_violin * 0.9,
                          showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(parts['bodies'], fill_colors):
        body.set_facecolor(color)
        body.set_edgecolor('black')
        body.set_alpha(0.7)
        body.set_linewidth(0.5)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.0)
    ax.set_xticks(range(n_elements))
    ax.set_xticklabels([Z_to_element.get(z, f'Z={z}') for z in elements_to_show])
    ax.set_xlabel('Element')
    ax.set_ylabel(r'$|F_\alpha^{\mathrm{DFT}} - F_\alpha^{\mathrm{Pred}}|$ (eV/Å)')
    legend_handles = [Patch(facecolor=model_colors[n], edgecolor='black',
                            alpha=0.7, label=n) for n in model_names]
    ax.legend(handles=legend_handles, loc='upper left', frameon=True)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.replace('.svg', '.png'), dpi=200, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.show()


def print_percentile_table(errors_by_model):
    all_elems = set()
    for per_elem in errors_by_model.values():
        all_elems.update(per_elem.keys())
    elements_to_show = sorted(all_elems)
    print(f"{'Element':<8}{'Model':<14}{'N':>8}{'mean':>10}{'p50':>10}{'p90':>10}{'p95':>10}{'p99':>10}{'max':>10}")
    print("-" * 88)
    for z in elements_to_show:
        sym = Z_to_element.get(z, f'Z={z}')
        for model_name, per_elem in errors_by_model.items():
            errs = per_elem.get(z, np.array([]))
            if len(errs) == 0:
                continue
            print(f"{sym:<8}{model_name:<14}{len(errs):>8}"
                  f"{np.mean(errs):>10.4f}"
                  f"{np.percentile(errs, 50):>10.4f}"
                  f"{np.percentile(errs, 90):>10.4f}"
                  f"{np.percentile(errs, 95):>10.4f}"
                  f"{np.percentile(errs, 99):>10.4f}"
                  f"{np.max(errs):>10.4f}")
def main():
    parser = argparse.ArgumentParser(
        description="Self-evaluation force-error distributions for two "
                    "(trajectory, model) pairs"
    )
    parser.add_argument('--traj1', required=True)
    parser.add_argument('--traj2', required=True)
    parser.add_argument('--model1', required=True)
    parser.add_argument('--model2', required=True)
    parser.add_argument('--label1', default='Model 1')
    parser.add_argument('--label2', default='Model 2')
    parser.add_argument('--output', default='violins_self_eval.svg')
    parser.add_argument('--log-scale', action='store_true')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    errors_by_model = {
        args.label1: evaluate_self(args.traj1, args.model1, args.label1, device),
        args.label2: evaluate_self(args.traj2, args.model2, args.label2, device),
    }
    plot_violins(errors_by_model, args.output, log_scale=args.log_scale)
    print_percentile_table(errors_by_model)
if __name__ == '__main__':
    main()