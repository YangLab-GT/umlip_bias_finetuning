import argparse
import numpy as np
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from ase.io import read, write
from matplotlib.cm import get_cmap
from dscribe.descriptors import SOAP
from sklearn.decomposition import PCA
# Plot style
plt.style.use('default')
plt.rc('font', size=18)
plt.rc('axes', titlesize=22)
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=15)
plt.rcParams['axes.linewidth'] = 3
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

def main():
    args = parse_arguments()
    print("==== Reading Data ====")
    ref_dataset = read(args.req1, index=":")
    eval_dataset = read(args.req2, index=args.splice)
    n_atoms = len(eval_dataset[0])
    print(f"Number of atoms: {n_atoms}")
    species = get_species(ref_dataset, eval_dataset)
    print(f"Species present: {species}")
    print("==== Converting to SOAP Descriptors ====")
    soap = define_soap(args, species)
    print("==== SOAP defined ====")
    ref_descriptors = generate_descriptors(soap, ref_dataset)
    print("==== Reference SOAP generated ====")
    eval_descriptors = generate_descriptors(soap, eval_dataset)
    print("==== Evaluation SOAP generated ====")
    print("==== Fitting PCA ====")
    pca = fit_pca(args.pca_components, ref_descriptors)
    if (args.plot == True):
        print("==== Generating Color Map ====")
        mapping, key = get_mapping_to_color(eval_dataset)
        type_to_element = {v: Z_to_element[k] for k, v in key.items()}
        print("==== Plotting ====")
        summary = plot_q_residuals(pca, eval_descriptors, n_atoms, mapping, type_to_element, args.figname)
    
    print("==== Q-Residual Summary per Atom Type ====") # gpt generated summary
    print(f"{'Type':>6} | {'N_atoms':>8} | {'Mean Q':>10}")
    print("-" * 30)
    for t in sorted(summary.keys()):
        n_atoms = summary[t]['n_atoms']
        mean_q = summary[t]['mean_q']
        print(f"{type_to_element[t]:>6} | {n_atoms:>8} | {mean_q:>10.3f}")
    print("-" * 30)
    print("==== Done ====")

def get_mapping_to_color(eval_dataset):
    """ Takes in atom numbers, returns mappings for colors
    Args:
        eval_dataset ([ase.Atoms]):
            Evaluation data
    Returns:
        numpy.ndarray(int):
            Atom types mapped to ordered integers
        dict[str, int]:
            "atom type": integer

    """
    unique_vals = np.unique(eval_dataset[0].arrays['numbers'])
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    remapped = np.array([mapping[v] for v in eval_dataset[0].arrays['numbers']])
    return remapped, mapping

def plot_q_residuals(pca, eval_descriptors, n_atoms, mapping, type_to_element, figname):
    """ Generates a plot of Q-residuals for each atom type over the trajectory,
    collects the mean and number of atoms per type.

    Args:
        pca (sklearn.decomposition.PCA):
            PCA model fit to reference descriptors
        eval_descriptors (numpy.ndarray):
            Generated evaluation descriptors
        n_atoms (int):
            Number of atoms in trajectory
        mapping (np.ndarray):
            Mapping of atom types to integers
        figname (str):
            Base name for figure files

    Returns:
        dict: {type_id: {'mean_q': float, 'n_atoms': int}}
    """
    unique_types = np.unique(mapping)
    cmap = plt.cm.get_cmap('viridis', len(unique_types))
    colors = [cmap(i) for i in range(len(unique_types))]
    summary = {}
    for t in unique_types:
        plt.figure()
        atom_indices = np.where(mapping == t)[0]
        q_residuals_all = []
        for i in atom_indices:
            residuals = compute_q_residuals(pca, eval_descriptors[i::n_atoms])
            q_residuals_all.extend(residuals)
            plt.plot(residuals, color=colors[t], alpha=0.5)
        q_residuals_all = np.array(q_residuals_all)
        mean_q = np.mean(q_residuals_all)
        n_atoms_type = len(atom_indices)
        summary[t] = {'mean_q': mean_q, 'n_atoms': n_atoms_type}
        plt.title(f"Atom Type {type_to_element[t]} (n={n_atoms_type}, mean Q={mean_q:.2f})")
        plt.savefig(f"{figname}_type_{type_to_element[t]}.svg")
        plt.close()
    return summary

def compute_q_residuals(pca, eval_descriptors):
    """ Computes Q-residuals
    Args: 
        pca (sklearn.decomposition.PCA):
            PCA model fit to reference descriptors
        eval_descriptors (numpy.ndarray):
            generated evaluation descriptors
    Returns:
        numpy.ndarray: 
            1D array of Q-residuals for each sample, 
            defined as the sum of squared reconstruction errors.
    """
    scores = pca.transform(eval_descriptors)
    X_reconstructed = np.dot(scores, pca.components_) + pca.mean_
    residuals = eval_descriptors - X_reconstructed
    q_residuals = np.sum(residuals**2, axis=1)
    return q_residuals
def fit_pca(n_components, ref_descriptors):
    """ Fits a PCA with set number of components to the reference dataset descriptors
    Args:
        n_components (int): 
            number of principle components to fit
        ref_descriptors (numpy.ndarray): 
            generated reference descriptors
    Returns:
        sklearn.decomposition.PCA: 
            a fit PCA model to reference SOAP descriptors
    """
    pca = PCA(n_components=n_components)
    pca.fit(ref_descriptors)
    return pca
def generate_descriptors(soap, atoms_dataset):
    """ Generates SOAP descriptors for a given dataset of atom structures
    Args:
        soap (dscribe.descriptors.SOAP): 
            A configured SOAP object ready to compute descriptors.
        atoms_dataset ([ase.Atoms]):
            A dataset/list of atomic structures
            for which SOAP features will be calculated.
    Returns:
        numpy.ndarray:
            A 2D array of shape (N_atoms_total, descriptor_length) where:
                - Each row corresponds to one atomic environment
                - N_atoms_total is the total sum of atoms across all structures
                - descriptor_length is the size of one SOAP feature vector
    """
    descriptors = [soap.create(atoms, centers=atoms.get_positions()) for atoms in atoms_dataset]
    descriptors_np =  np.array(descriptors).reshape(-1, descriptors[0].shape[-1])
    return descriptors_np
def get_species(ref_dataset, eval_dataset):
    """ Get atom types from datasets
    Args:
        ref_dataset  ([ase.Atoms]):
            Reference data
        eval_dataset ([ase.Atoms]):
            Evaluation data
    Returns:
        [str]:
            Unique atom types present in data
    """
    species = set()
    for atoms in (ref_dataset):
        species.update(atoms.get_chemical_symbols())
    for atoms in (eval_dataset):
        species.update(atoms.get_chemical_symbols())
    species = list(species)
    return species
def define_soap(args, species):
    """ Define SOAP parameters object
    Args:
        args     (argparse.Namespace):
            input arguments
        species  ([str]):
            Unique atom types present in data
    returns
        dscribe.descriptors.SOAP:
            soap descriptor object
    """
    soap = SOAP(
        species=species,
        r_cut=args.rcut,
        n_max=args.nmax,
        l_max=args.lmax,
        sigma=args.sigma,
        periodic=args.periodic
    )
    return soap
def parse_arguments():
    """ Parses command line arguments.
    Returns:
        argparse object with:
            req1 (str): path to dataset
            req2 (str): path to evaluation
            optional arguments:
                --rcut     (float): radial cutoff.   default: 5.0   (SOAP)
                --nmax     (int): radial basis size. default: 8     (SOAP)
                --lmax     (int): angular momentum.  default: 6     (SOAP)
                --sigma    (float): gaussian smear.  default: 0.5   (SOAP)
                --periodic (bool): periodicity.      default: False (SOAP)
                --splice   (str):  eval traj splice. default: :     (input)
                --splice   (str):  PCA components.   default: 5     (pca)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("req1") 
    parser.add_argument("req2") 
    parser.add_argument("--rcut", type=float, default=5.0, help="Radial cutoff (default: 5.0)")
    parser.add_argument("--nmax", type=int, default=8, help="Radial basis size (default: 8)")
    parser.add_argument("--lmax", type=int, default=6, help="Maximum angular momentum (default: 6)")
    parser.add_argument("--sigma", type=float, default=0.375, help="Gaussian smearing (default: 0.375)")
    parser.add_argument("--periodic", action="store_false", help="Enable periodic mode (default: True)")
    parser.add_argument("--splice", type=str, default="::100", help="Define splicing for evaluation trajectory (default: ::100)")
    parser.add_argument("--pca_components", type=int, default=5, help="Define number of PC components (default: 5)")
    parser.add_argument("--plot", action="store_false", help="Plot Q-residuals (default: True)")
    parser.add_argument("--figname", type=str, default="qresiduals", help="Title of plotted figure (defualt: qresiduals)")
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    main()
