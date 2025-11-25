import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import sys
import json
from pathlib import Path
from sklearn.metrics import root_mean_squared_error
from ase import units
from ase.md.langevin import Langevin
from ase.io import read, write
import numpy as np
from mace.calculators import MACECalculator
from pymatgen.io.vasp.outputs import Vasprun
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from pymatgen.core import Structure
"""First command line argument = directory"""
"""Second CLA = model"""
def main():
    calc = MACECalculator(sys.argv[2])
    pairs = get_respective_files(sys.argv[1])
    dft_matrix, mace_matrix, error_matrix, timesteps, mace_stress, dft_stress, dft_energy, mace_energy  = compute_all(pairs, calc)
    # plot_force_parity_components(dft_matrix, mace_matrix, dist_from_interesting)
    # print(compute_RMSE_and_plot(dft_matrix, mace_matrix, timesteps))
    print(compute_stress_RMSE(dft_stress, mace_stress))
    print(compute_RMSE(dft_matrix, mace_matrix))
    print(compute_energy_RMSE(dft_energy, mace_energy))
    # rmse_by_frame = compute_rmse_per_frame(dft_matrix, mace_matrix, timesteps)
    # plot_rmse_by_frame(rmse_by_frame, timesteps)
    # for i in range(1,197):
    #     plot_forces_by_frame(dft_matrix, mace_matrix, i, timesteps)
    #     plot_forces_by_frame_1x3(dft_matrix, mace_matrix, i, timesteps)
    #     calculate_energy_drift(dft_matrix, mace_matrix, timesteps, i)
def compute_energy_RMSE(dft, mace):
    return np.sqrt(np.mean((dft-mace)**2))
def compute_stress_RMSE(dft, mace):
    return np.sqrt(np.mean((dft - mace)**2))
def compute_RMSE(dft_matrix, mace_matrix):
    """Takes in DFT / Predicted Matrices of form
    Params:
        dft_matrix, pred matrix = [[[x1, y1, z1, fx1, fy1, fz1],
                                    [x2, y2, z2, fx2, fy2, fz2],
                                    ...
                                    [xN, yN, zN, fxN, fyN, fzN]],...]
    Returns:
        fx, fy, fz
    """
    dft_forces = dft_matrix[:, :, 3:]
    mace_forces = mace_matrix[:, :, 3:]
    rmse = {}
    directions = ['fx', 'fy', 'fz']
    for i, dir in enumerate(directions):
        error = dft_forces[:, :, i] - mace_forces[:, :, i]
        mse = np.mean(error ** 2)
        rmse[dir] = np.sqrt(mse)
    return rmse
import numpy as np
import matplotlib.pyplot as plt

def compute_RMSE_and_plot(dft_matrix, mace_matrix, timesteps):
    """
    Computes per-axis RMSE, overall RMSE per timestep, plots histogram,
    and finds frame with maximum single-atom force error.

    Params:
        dft_matrix, mace_matrix: arrays (n_frames, n_atoms, 6)
        timesteps: list or array of frame indices / times (len = n_frames)
    """
    dft_forces = dft_matrix[:, :, 3:]
    mace_forces = mace_matrix[:, :, 3:]
    

    n_frames = dft_matrix.shape[0]
    n_atoms = dft_matrix.shape[1]
    max_forces_dft = []
    max_forces_mace = []

    for frame in range(n_frames):
        # Extract per-atom forces (fx, fy, fz)
        dft_forces_c = dft_matrix[frame, :, 3:6]  # shape (n_atoms,3)
        mace_forces_c = mace_matrix[frame, :, 3:6]

        # Compute magnitude of each atom's force
        dft_mag = np.linalg.norm(dft_forces_c, axis=1)  # shape (n_atoms,)
        mace_mag = np.linalg.norm(mace_forces_c, axis=1)

        # Take maximum per frame
        max_forces_dft.append(np.max(dft_mag))
        max_forces_mace.append(np.max(mace_mag))
            # Convert to arrays
    max_forces_dft = np.array(max_forces_dft)
    max_forces_mace = np.array(max_forces_mace)

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(timesteps, max_forces_dft, 'o', label='DFT max force')
    plt.plot(timesteps, max_forces_mace, 'o', label='MACE max force')
    plt.xlabel('Frame')
    plt.ylabel('Max force magnitude (eV/Å)')
    plt.title('Maximum force per frame')
    plt.legend()
    plt.savefig("results_maxForcePerFrame.svg")
    # === Global per-axis RMSE ===
    rmse = {}
    directions = ['fx', 'fy', 'fz']
    for i, dir in enumerate(directions):
        error = dft_forces[:, :, i] - mace_forces[:, :, i]
        mse = np.mean(error ** 2)
        rmse[dir] = np.sqrt(mse)
    # === Overall force magnitude errors ===
    force_diff = dft_forces - mace_forces
    mag_error = np.linalg.norm(force_diff, axis=2)
    
    # === Per-timestep RMSE ===
    timestep_rmse = []
    for i in range(dft_forces.shape[0]):
        frame_err = dft_forces[i] - mace_forces[i]
        mse_frame = np.mean(frame_err ** 2)
        rmse_frame = np.sqrt(mse_frame)
        timestep_rmse.append(rmse_frame)
        # print(f"timestep {timesteps[i]} → RMSE = {rmse_frame:.4f} eV/Å")

    timestep_rmse = np.array(timestep_rmse)
    
    # # === Find location of maximum error ===
    # max_idx = np.unravel_index(np.argmax(mag_error), mag_error.shape)
    # max_conf, max_atom = max_idx
    # max_val = mag_error[max_conf, max_atom]
    
    # print("\n🔍 Largest single-atom force error:")
    # print(f"   Magnitude: {max_val:.3f} eV/Å")
    # print(f"   Frame (timestep): {timesteps[max_conf]}")
    # print(f"   Atom index: {max_atom}")
    # print(f"   DFT force:  {dft_forces[max_conf, max_atom]}")
    # print(f"   MACE force: {mace_forces[max_conf, max_atom]}")
    
    # === Plot histogram of all force magnitude errors ===
    plt.figure(figsize=(7,5))
    plt.hist(mag_error.flatten(), bins=50, edgecolor='black')
    plt.title("Distribution of Force Errors (‖ΔF‖)")
    plt.xlabel("Force Error Magnitude (eV/Å)")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("Test_naive.svg")
    plt.show()
    
    # === Plot RMSE vs timestep ===
    plt.figure(figsize=(7,5))
    plt.plot(timesteps, timestep_rmse, 'o')
    plt.title("RMSE of Forces per Timestep")
    plt.xlabel("Timestep")
    plt.ylabel("RMSE (eV/Å)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("RMSE_vs_timestep_periodic.svg")
    plt.show()
    
    return rmse
def get_respective_files(root_dir):
    """Takes in directory, returns all pairs of POSCAR/vasprun.
    Params:
        Root directory
    Returns:
        matrix of form: [[.../POSCAR, .../vasprun.xml], ...]
    """
    poscars = {}
    vaspruns = {}
    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in filenames:
            if filename == "POSCAR":
                parent = os.path.basename(dirpath)
                poscars[parent] = os.path.join(dirpath, filename)
            elif filename == "vasp_info.json":
                parent = os.path.basename(dirpath)
                # write_info_to_file(os.path.join(dirpath, filename), os.path.join(dirpath, "vasp_info.json"))
                vaspruns[parent] = os.path.join(dirpath, filename)
    paired = []
    for key in poscars:
        if key in vaspruns:
            paired.append((poscars[key], vaspruns[key]))
    return paired
def compute_all(pairs, mace_calc):
    """Takes in all pairs of POSCAR/vasprun, computes error
    Params:
        pairs: [POSCAR/vasprun] pairs
        mace_calc: MACECalculator object
    Returns:
        dft_matrix:
        mace_matrix:
        error_matrix: dft_matrix-mace_matrix
        dist_from_interesting:
    """
    dft_matrix, mace_matrix, error_matrix, mace_stress_arr, dft_stress_arr, mace_energies, dft_energies = [], [], [], [], [], [], []
    dist_from_interesting = []
    total = len(pairs)
    for i, pair in enumerate(pairs, start=1):
        timestep = extract_timestep(pair[0])
        dft, mace, error, converged, mace_stress, dft_stress, mace_energy, dft_energy = compute_error(pair, mace_calc)
        if(converged):
            dft_energies.append(dft_energy*1000/197)
            mace_energies.append(mace_energy*1000/197)
            dft_stress_gpa = dft_stress.flatten() / (1602.1766208 * -1)
            dft_stress_gpa_voigt = np.array([dft_stress_gpa[0], # xx
                                      dft_stress_gpa[4], # yy
                                      dft_stress_gpa[8], # zz
                                      dft_stress_gpa[1], # yz
                                      dft_stress_gpa[2], # xz
                                      dft_stress_gpa[3]]) # xy
            dist_from_interesting.append(timestep-0)
            dft_matrix.append(dft)
            mace_matrix.append(mace)
            # print(mace)
            error_matrix.append(error)
            mace_stress_arr.append(mace_stress)
            dft_stress_arr.append(dft_stress_gpa_voigt)
        percent_done = (i / total) * 100
        print(f"Progress: {percent_done:.2f}% ({i}/{total})")
    dft_energies = np.array(dft_energies)
    mace_energies = np.array(mace_energies)
    dft_matrix = np.array(dft_matrix)
    mace_matrix = np.array(mace_matrix)
    error_matrix = np.array(error_matrix)
    mace_stress_matrix = np.array(mace_stress_arr)
    dft_stress_matrix = np.array(dft_stress_arr)
    return dft_matrix, mace_matrix, error_matrix, dist_from_interesting, mace_stress_matrix, dft_stress_matrix, dft_energies, mace_energies
def compute_error(pair, mace_calc):
    """Takes in pair of POSCAR, vasprun, computes forces using MACE / collects forces from vasprun, computes error"""
    """If vr is unconverged, skip this"""
    poscar_path = pair[0]
    vr_path = pair[1]
    dft_force, converged, dft_stress, dft_energy = get_forces_from_dft(vr_path)
    if(converged == False):
        return None, None, None, False, None, None, None, None
    else:
        mace_force, mace_stress, mace_energy = compute_forces(poscar_path, mace_calc)
    return dft_force, mace_force, (dft_force-mace_force), converged, mace_stress, dft_stress, mace_energy, dft_energy
def compute_forces(poscar_file, mace_calc):
    """Takes in POSCAR, model, computes forces"""
    atoms = read(poscar_file)
    #print(atoms[79])
    atoms.calc = mace_calc
    mace_positions = atoms.get_positions()
    mace_forces = atoms.get_forces()
    mace_stresses = atoms.get_stress()
    mace_energy = atoms.get_potential_energy()
    return np.hstack((mace_positions, mace_forces)), mace_stresses, mace_energy
def get_forces_from_dft(json_path):
    """
    Takes in a JSON file from write_info_to_file / write_vasprun_json,
    returns (per_frame, success_flag, stress)
    
    per_frame: Nx6 array [[x,y,z,fx,fy,fz], ...]
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    try:
        # Reconstruct structure
        structure = Structure.from_dict(data["structure"])

        # Convert lists to NumPy arrays
        forces = np.array(data["forces"])
        stresses = np.array(data["stress"])
        energy = data["energy"]
        per_frame = []
        for site, force in zip(structure.sites, forces):
            row = np.concatenate([site.coords, force])
            per_frame.append(row)
        per_frame = np.array(per_frame)
        return per_frame, True, stresses, energy

    except Exception as e:
        print(f"Error reading JSON: {e}")
        return None, False, None, None
def plot_force_parity_components(dft_matrix, mace_matrix, dist_from_interesting, title_prefix="DFT vs MACE Force Parity"):
    """
    GPT Generated
    Creates three parity plots: fx, fy, fz.
    Colors each point by its structure's dist_from_interesting value.
    """
    # Extract force components
    dft_forces = dft_matrix[:, :, 3:]  # fx, fy, fz
    mace_forces = mace_matrix[:, :, 3:]

    # Flatten each component
    dft_fx = dft_forces[:, :, 0].flatten()
    dft_fy = dft_forces[:, :, 1].flatten()
    dft_fz = dft_forces[:, :, 2].flatten()

    mace_fx = mace_forces[:, :, 0].flatten()
    mace_fy = mace_forces[:, :, 1].flatten()
    mace_fz = mace_forces[:, :, 2].flatten()

    # Prepare color mapping
    cmap = get_cmap("seismic")
    norm = Normalize(vmin=min(dist_from_interesting), vmax=max(dist_from_interesting))

    # Repeat each structure's color for all its atoms
    n_atoms = dft_matrix.shape[1]
    color_values = np.repeat(dist_from_interesting, n_atoms)
    colors = cmap(norm(color_values))

    # Plot each component
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    components = ['fx', 'fy', 'fz']
    dft_all = [dft_fx, dft_fy, dft_fz]
    mace_all = [mace_fx, mace_fy, mace_fz]

    for i in range(3):
        ax = axes[i]
        sc = ax.scatter(dft_all[i], mace_all[i], c=color_values, cmap=cmap, norm=norm, s=5, alpha=0.6, edgecolors='none')
        min_val = min(dft_all[i].min(), mace_all[i].min())
        max_val = max(dft_all[i].max(), mace_all[i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal parity')
        ax.set_xlabel(f"DFT {components[i]} (eV/Å)")
        ax.set_ylabel(f"MACE {components[i]} (eV/Å)")
        ax.set_title(f"{title_prefix}: {components[i]}")
        ax.legend()
        ax.grid(True)

    # Add colorbar
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), orientation='vertical')
    cbar.set_label("Distance from Interesting")

    # plt.tight_layout()
    plt.savefig("parity_plot_colored.png")
    plt.show()
def write_info_to_file(vasprun_path, output_file="vasp_info.json"):
    # if os.path.exists(output_file):
    #     print(f"JSON file already exists: {output_file}, skipping.")
    #     return
    vr = Vasprun(vasprun_path)
    if vr.as_dict().get('has_vasp_completed', False):
        step0 = vr.ionic_steps[0]
        structure = step0['structure']
        forces = step0['forces']
        stresses = step0['stress']
        energy = step0['e_0_energy']
        struct_dict = structure.as_dict()
        data = {
            "structure": struct_dict,
            "forces": forces.tolist(),
            "stress": stresses.tolist(),
            "energy": energy
        }
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Data written to {output_file}")
    else:
        print("VASP calculation did not complete successfully.")
def extract_timestep(path):
    """
    Extracts the timestep from a path like .../naive.x/POSCAR
    Returns it as an integer.
    """
    parent = os.path.basename(os.path.dirname(path))
    timestep = int(parent.split('.')[-1])
    return timestep
def compute_rmse_per_frame(dft_matrix, mace_matrix, timesteps):
    """
    Computes RMSE per frame for each direction.
    Returns a dictionary with lists of RMSEs for fx, fy, fz.
    """
    directions = ['fx', 'fy', 'fz']
    rmse_by_frame = {dir: [] for dir in directions}
    for frame in range(dft_matrix.shape[0]):
        dft_forces = dft_matrix[frame, :, 3:]
        mace_forces = mace_matrix[frame, :, 3:]
        for i, dir in enumerate(directions):
            error = dft_forces[:, i] - mace_forces[:, i]
            #error = dft_forces[i] - mace_forces[i]
            # print(error)
            mse = np.mean(error ** 2)
            rmse_by_frame[dir].append(mse)
    return rmse_by_frame
def plot_rmse_by_frame(rmse_by_frame, timesteps, title="Per-Frame RMSE by Direction"):
    """
    Plots RMSE vs timestep for fx, fy, fz.
    """
    plt.figure(figsize=(10, 6))
    for dir, rmse_values in rmse_by_frame.items():
        plt.plot(timesteps, rmse_values, 'o', label=f"{dir} error")

    plt.xlabel("Timestep")
    plt.ylabel("Error (eV/Å)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    # plt.tight_layout()
    plt.savefig("error_by_frame_total.png")
    plt.show()
import os
import matplotlib.pyplot as plt

def plot_forces_by_frame_1x3(dft_matrix, mace_matrix, atom_id, timesteps):
    colors = ['#004488', "#BB5566", "#DDAA33", "#228833", "#7744AA", "#66CCEE"]

    plt.rc('font', size=14)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=16)
    plt.rc('xtick', labelsize=11)
    plt.rc('ytick', labelsize=11)
    plt.rc('legend', fontsize=11)
    plt.rc('figure', titlesize=18)

    directions = ['fx', 'fy', 'fz']

    # Create single figure with 3 horizontally aligned subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=True)
    timesteps = np.array(timesteps)
    for i, direction in enumerate(directions):
        ax = axes[i]

        dft_forces = dft_matrix[:, atom_id, 3 + i]
        md_forces  = mace_matrix[:, atom_id, 3 + i]
    
        # Plot
        ax.plot(timesteps/2000, dft_forces, 'o', alpha=0.8, color=colors[1], label='DFT')
        ax.plot(timesteps/2000, md_forces, 's', alpha=0.8, color=colors[0], label='MD (MACE)')

        # Labels only (no titles)
        
        ax.set_ylabel(f"Force {direction} (eV/Å)")
        ax.grid(True)
        # Only first subplot shows legend
        if i == 2:
            ax.legend(loc='upper right')
    axes[0].set_xlabel("Time (ns)")
    axes[1].set_xlabel("Time (ns)")
    axes[2].set_xlabel("Time (ns)")
    plt.tight_layout()

    # Saving
    results_dir = f"./force_plots/{atom_id}/"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    plt.savefig(f"{results_dir}/forces_1x3.png")
    plt.close()

def plot_forces_by_frame(dft_matrix, mace_matrix, atom_id, timesteps):
    colors = ['#004488', "#BB5566", "#DDAA33", "#228833", "#7744AA", "#66CCEE"]
    plt.rc('font', size=14)          # default text sizes
    plt.rc('axes', titlesize=16)     # axes title
    plt.rc('axes', labelsize=16)     # x and y labels
    plt.rc('xtick', labelsize=11)    
    plt.rc('ytick', labelsize=11)    
    plt.rc('legend', fontsize=11)    
    plt.rc('figure', titlesize=18)  
    directions = ['fx', 'fy', 'fz']
    for i, direction in enumerate(directions):
        dft_forces = dft_matrix[:, atom_id, 3 + i]
        md_forces  = mace_matrix[:, atom_id, 3 + i]
        plt.figure(figsize=(8, 5))
        plt.plot(timesteps, dft_forces, 'o', label='DFT', alpha=0.8, color=colors[1])
        plt.plot(timesteps, md_forces, 's', label='MD (MACE)', alpha=0.8, color=colors[0])
        plt.xlabel("Timestep")
        plt.ylabel(f"Force {direction} (eV/Å)")
        # plt.title(f"Force Comparison for Atom {atom_id} — {direction}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        results_dir = f"./force_plots/{atom_id}/"
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        plt.savefig(f"./force_plots/{atom_id}/force_by_frame_{direction}.png")
        plt.close()
def calculate_energy_drift(dft_matrix, mace_matrix, timesteps, specific_atom = None):
    """ Here we compute the integration of force errors over time to see energy drift """
    timesteps = np.array(timesteps)
    sort_idx = np.argsort(timesteps)
    dft_sorted = dft_matrix[sort_idx]
    mace_sorted = mace_matrix[sort_idx]
    timesteps_sorted = timesteps[sort_idx]
    # isolate so easier to deal with...
    positions = dft_sorted[:, :, 0:3]
    forces_dft = dft_sorted[:, :, 3:6]
    forces_mace = mace_sorted[:, :, 3:6]
    if(specific_atom is not None):
        positions_s = dft_sorted[:, specific_atom, 0:3]
        forces_dft_s = dft_sorted[:, specific_atom, 3:6]
        forces_mace_s = mace_sorted[:, specific_atom, 3:6]
        # take difference 
        displacements_s = positions_s[1:] - positions_s[:-1]
        force_errors_s = forces_mace_s[:-1] - forces_dft_s[:-1]
        # Compute change in energy
        delta_U_s = -np.sum(force_errors_s * displacements_s, axis=(1))
        cumulative_energy_drift_s = np.cumsum(delta_U_s)
        # Plot energy error vs actual time
        plt.figure(figsize=(8,5))
        plt.plot(timesteps_sorted[1:], delta_U_s, marker='o')
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative Energy Error (approx.)")
        plt.title(f"Energy Error over Time - atom {specific_atom}")
        plt.grid(True)
        plt.savefig(f"./force_plots/{specific_atom}/energy_drift_{specific_atom}.svg")
        plt.show()
    # take difference 
    displacements = positions[1:] - positions[:-1]
    force_errors = forces_mace[:-1] - forces_dft[:-1]
    # Compute change in energy
    delta_U = -np.sum(force_errors * displacements, axis=(1,2))
    cumulative_energy_drift = np.cumsum(delta_U)

    # Plot energy error vs actual time
    plt.figure(figsize=(8,5))
    plt.plot(timesteps_sorted[1:], delta_U, marker='o')
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Energy Error (approx.)")
    plt.title("Energy Error over Time")
    plt.grid(True)
    plt.savefig("energy_drift.svg")
    plt.show()
    return cumulative_energy_drift
if __name__ == "__main__":
    main()
