# -*- coding: utf-8 -*-
"""
Data pre-processing to provide chemical descriptors.

This script executes the conversion of chemical structures into numerical data using RDkit 2D descriptors.
Dataset: chemical structures (SMILES) and the corresponding properties
Libraries: Pandas and RDkit

*SMILES = Simplified Molecular Input Line Entry System

Author: 
    Adroit T.N. Fajar, Ph.D. (Dr.Eng.)
    Scopus Author ID: 57192386143
    ResearcherID: HNI-7382-2023
    ResearchGate: https://researchgate.net/profile/Adroit-Fajar
    GitHub: https://github.com/adroitfajar

"""

### load dataset
import pandas as pd
dataset = pd.read_csv('fus_ent_data.csv') # this contains SMILES and specific properties of the chemicals

### add a column of molecule descriptions
from rdkit.Chem import PandasTools
PandasTools.AddMoleculeColumnToFrame(dataset, smilesCol='smiles') #! flags

### define the list of molecules
molecules = dataset['ROMol']

### briefly checking the structures
PandasTools.FrameToGridImage(dataset.head(20), legendsCol='fusH', molsPerRow=4)

### list of RDkit 2D descriptors
from rdkit.Chem import Descriptors
desc_list = [n[0] for n in Descriptors._descList]
print(len(desc_list))
print(desc_list)

### calculate descriptors for each molecule
from rdkit.ML.Descriptors import MoleculeDescriptors
calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)
rdkit_desc = [calc.CalcDescriptors(m) for m in molecules] #! flags

### export data into a csv file
results = pd.DataFrame(rdkit_desc, index=dataset.fusH, columns=desc_list)
filtered_res = results.dropna() # ensure to drop strange results indicated by NaN
remove = [133.2, 135.5, 149.6, 162.4, 176.0, 193.2] # these data are outliers
processed_res = filtered_res.drop(index = remove) # ensure to drop outliers
processed_res.to_csv('fus_ent_desc.csv')