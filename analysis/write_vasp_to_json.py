import os
import sys
import json
import numpy as np
from pymatgen.io.vasp.outputs import Vasprun
"""First command line argument = directory"""
def main():
    delete = False
    if (len(sys.argv) >= 3):
        delete = bool(sys.argv[2])
    pairs = get_respective_files(sys.argv[1], delete)
def get_respective_files(root_dir, delete = False):
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
            elif filename == "vasprun.xml":
                parent = os.path.basename(dirpath)
                write_info_to_file(os.path.join(dirpath, filename), os.path.join(dirpath, "vasp_info.json"))
            files_to_remove = ["CHG", "CHGCAR", "CONTCAR", "DOSCAR", "EIGENVAL", "IBZKPT", "PCDAT", "PROCAR"]
            for file in files_to_remove:
                if filename == file and delete == True:
                    file_to_delete = os.path.join(dirpath, filename)
                    os.remove(file_to_delete)
                    print(f"File '{file_to_delete}' deleted.")

    paired = []
    for key in poscars:
        if key in vaspruns:
            paired.append((poscars[key], vaspruns[key]))
    return paired
def write_info_to_file(vasprun_path, output_file="vasp_info.json"):
    """ Writes vasprun.xml to json file 
    """
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
if __name__ == "__main__":
    main()