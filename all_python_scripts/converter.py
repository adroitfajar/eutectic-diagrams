# -*- coding: utf-8 -*-
"""
This script is to convert chemical names into their SMILES codes.
*SMILES = Simplified Molecular Input Line Entry System
Dataset: list of chemical names
Libraries: CirPy, Pandas, Numpy

NOTE!!!
If there are strange results, confirm the SMILES codes using ChemDraw.

Author: 
    Adroit T.N. Fajar, Ph.D. (Dr.Eng.)
    Scopus Author ID: 57192386143
    ResearcherID: HNI-7382-2023
    ResearchGate: https://researchgate.net/profile/Adroit-Fajar
    GitHub: https://github.com/adroitfajar

"""

import cirpy
import pandas as pd
import numpy as np

data = pd.read_excel("fus_ent_name.xlsx", sheet_name = "Sheet1")
names = []
identifiers = data["Name"].values

for name in identifiers:
    names += [name, cirpy.resolve(name, 'smiles')]

New = np.array(names).reshape(-1, 2)
PD = pd.DataFrame(New)
PD.to_excel("smiles.xlsx")