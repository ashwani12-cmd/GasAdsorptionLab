#!/usr/bin/env python3

"""
Generate the GasAdsorptionLab molecule library.

Priority:
1. ASE built-in molecule database
2. Manual definitions (for unsupported molecules)

Output:
    gal/data/gases/*.xyz
"""

from pathlib import Path
from ase.build import molecule
from ase.io import write
from ase import Atoms

OUTPUT = Path("gal/data/gases")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Complete Gas Library
# ==========================================================

GASES = [

    # Atmospheric
    "H2","N2","O2","H2O","O3",

    # Environmental
    "CO","CO2","NO","NO2","N2O",
    "NH3","SO2","SO3",

    # Toxic
    "H2S","HCN","PH3","HF","HCl","Cl2",

    # Hydrocarbons
    "CH4","C2H2","C2H4","C2H6","C3H6","C3H8",

    # VOCs
    "CH3OH","C2H5OH","CH2O",
    "CH3CHO","Acetone",
    "Benzene","Toluene",

    # Sulfur
    "CS2","COS",

    # Semiconductor
    "SiH4","GeH4","B2H6","AsH3",

    # Fluorinated
    "CF4","CHF3","C2F6","SF6","NF3",

    # Noble gases
    "He","Ne","Ar","Kr","Xe",
]

# ==========================================================
# ASE name mapping
# ==========================================================

ASE_NAMES = {

    "H2":"H2",
    "N2":"N2",
    "O2":"O2",
    "H2O":"H2O",
    "O3":"O3",

    "CO":"CO",
    "CO2":"CO2",
    "NO":"NO",
    "NO2":"NO2",
    "N2O":"N2O",
    "NH3":"NH3",

    "SO2":"SO2",

    "CH4":"CH4",
    "C2H2":"C2H2",
    "C2H4":"C2H4",
    "C2H6":"C2H6",

    "CH3OH":"CH3OH",
    "C2H5OH":"CH3CH2OH",

    "CH2O":"CH2O",

    "HF":"HF",
    "HCl":"HCl",
    "Cl2":"Cl2",

    "Benzene":"C6H6",

    "SiH4":"SiH4",
}

# ==========================================================
# Manual molecules
# ==========================================================

MANUAL = {

"H2S":
Atoms(
    "SH2",
    positions=[
        [0.000,0.000,0.000],
        [0.958,0.000,0.000],
        [-0.240,0.927,0.000],
    ]
),

"PH3":
Atoms(
    "PH3",
    positions=[
        [0,0,0],
        [1.42,0,0.45],
        [-0.71,1.23,0.45],
        [-0.71,-1.23,0.45],
    ]
),

"He":Atoms("He"),
"Ne":Atoms("Ne"),
"Ar":Atoms("Ar"),
"Kr":Atoms("Kr"),
"Xe":Atoms("Xe"),
}

# ==========================================================
# Generation
# ==========================================================

success = []
failed = []

print("="*60)
print("Generating Gas Library")
print("="*60)

for gas in GASES:

    try:

        if gas in ASE_NAMES:

            atoms = molecule(ASE_NAMES[gas])

        elif gas in MANUAL:

            atoms = MANUAL[gas]

        else:

            raise ValueError("No definition available")

        outfile = OUTPUT/f"{gas}.xyz"

        write(outfile, atoms)

        success.append(gas)

        print(f"[✓] {gas:12s}")

    except Exception as e:

        failed.append(gas)

        print(f"[X] {gas:12s} {e}")

print("\nFinished")
print(f"Successful : {len(success)}")
print(f"Failed     : {len(failed)}")

if failed:

    print("\nNeed manual structures for:")

    for g in failed:
        print("  ", g)
