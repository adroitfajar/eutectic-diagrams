# -*- coding: utf-8 -*-
"""
This script executes estimation of activity coefficients using openCOSMO-RS.
DFT calculation is required before executing this script, to provide cosmo files.
Here, we use ORCA to run DFT calculations, generating *.orcacosmo files.
*openCOSMO-RS and ORCA were selected since they support open-source knowledge.

Author: 
    Adroit T.N. Fajar, Ph.D. (Dr.Eng.)
    Scopus Author ID: 57192386143
    ResearcherID: HNI-7382-2023
    ResearchGate: https://researchgate.net/profile/Adroit-Fajar
    GitHub: https://github.com/adroitfajar

"""

### import libraries
import numpy as np
from opencosmorspy import COSMORS

### choose a default parameterization (default_orca)
crs = COSMORS(par='default_orca')
crs.par.calculate_contact_statistics_molecule_properties = True
print(crs.par)

### load molecules (provide a list of paths where the *.orcacosmo files are located)
hba = (
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/trimethylphosphine_oxide/COSMO_TZVPD/trimethylphosphine_oxide_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/triethylphosphine_oxide/COSMO_TZVPD/triethylphosphine_oxide_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/tripropylphosphine_oxide/COSMO_TZVPD/tripropylphosphine_oxide_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/tributylphosphine_oxide/COSMO_TZVPD/tributylphosphine_oxide_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/tripentylphosphine_oxide/COSMO_TZVPD/tripentylphosphine_oxide_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/trihexylphosphine_oxide/COSMO_TZVPD/trihexylphosphine_oxide_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/triheptylphosphine_oxide/COSMO_TZVPD/triheptylphosphine_oxide_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/trioctylphosphine_oxide/COSMO_TZVPD/trioctylphosphine_oxide_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/trinonylphosphine_oxide/COSMO_TZVPD/trinonylphosphine_oxide_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/tris(decyl)phosphine_oxide/COSMO_TZVPD/tris(decyl)phosphine_oxide_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/(methylsulfinyl)methane/COSMO_TZVPD/(methylsulfinyl)methane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/(methylsulfinyl)ethane/COSMO_TZVPD/(methylsulfinyl)ethane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(methylsulfinyl)propane/COSMO_TZVPD/1-(methylsulfinyl)propane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(methylsulfinyl)butane/COSMO_TZVPD/1-(methylsulfinyl)butane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(methylsulfinyl)pentane/COSMO_TZVPD/1-(methylsulfinyl)pentane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(methylsulfinyl)hexane/COSMO_TZVPD/1-(methylsulfinyl)hexane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(methylsulfinyl)heptane/COSMO_TZVPD/1-(methylsulfinyl)heptane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(methylsulfinyl)octane/COSMO_TZVPD/1-(methylsulfinyl)octane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(methylsulfinyl)nonane/COSMO_TZVPD/1-(methylsulfinyl)nonane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(methylsulfinyl)decane/COSMO_TZVPD/1-(methylsulfinyl)decane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(methylsulfinyl)undecane/COSMO_TZVPD/1-(methylsulfinyl)undecane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(methylsulfinyl)dodecane/COSMO_TZVPD/1-(methylsulfinyl)dodecane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(methylsulfinyl)tridecane/COSMO_TZVPD/1-(methylsulfinyl)tridecane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/(ethylsulfinyl)ethane/COSMO_TZVPD/(ethylsulfinyl)ethane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(propylsulfinyl)propane/COSMO_TZVPD/1-(propylsulfinyl)propane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(butylsulfinyl)butane/COSMO_TZVPD/1-(butylsulfinyl)butane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(pentylsulfinyl)pentane/COSMO_TZVPD/1-(pentylsulfinyl)pentane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(hexylsulfinyl)hexane/COSMO_TZVPD/1-(hexylsulfinyl)hexane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(heptylsulfinyl)heptane/COSMO_TZVPD/1-(heptylsulfinyl)heptane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(octylsulfinyl)octane/COSMO_TZVPD/1-(octylsulfinyl)octane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(nonylsulfinyl)nonane/COSMO_TZVPD/1-(nonylsulfinyl)nonane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(decylsulfinyl)decane/COSMO_TZVPD/1-(decylsulfinyl)decane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(undecylsulfinyl)undecane/COSMO_TZVPD/1-(undecylsulfinyl)undecane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(dodecylsulfinyl)dodecane/COSMO_TZVPD/1-(dodecylsulfinyl)dodecane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-(tridecylsulfinyl)tridecane/COSMO_TZVPD/1-(tridecylsulfinyl)tridecane_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,1,3,3-tetramethylurea/COSMO_TZVPD/1,1,3,3-tetramethylurea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,3-diethyl-1,3-dimethylurea/COSMO_TZVPD/1,3-diethyl-1,3-dimethylurea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,1,3,3-tetraethylurea/COSMO_TZVPD/1,1,3,3-tetraethylurea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,3-diethyl-1,3-dipropylurea/COSMO_TZVPD/1,3-diethyl-1,3-dipropylurea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,1,3,3-tetrapropylurea/COSMO_TZVPD/1,1,3,3-tetrapropylurea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,3-dibutyl-1,3-dipropylurea/COSMO_TZVPD/1,3-dibutyl-1,3-dipropylurea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,1,3,3-tetrabutylurea/COSMO_TZVPD/1,1,3,3-tetrabutylurea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,3-dibutyl-1,3-dipentylurea/COSMO_TZVPD/1,3-dibutyl-1,3-dipentylurea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,1,3,3-tetrapentylurea/COSMO_TZVPD/1,1,3,3-tetrapentylurea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,3-dihexyl-1,3-dipentylurea/COSMO_TZVPD/1,3-dihexyl-1,3-dipentylurea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,1,3,3-tetrahexylurea/COSMO_TZVPD/1,1,3,3-tetrahexylurea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,3-diheptyl-1,3-dihexylurea/COSMO_TZVPD/1,3-diheptyl-1,3-dihexylurea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,1,3,3-tetramethylthiourea/COSMO_TZVPD/1,1,3,3-tetramethylthiourea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,3-diethyl-1,3-dimethylthiourea/COSMO_TZVPD/1,3-diethyl-1,3-dimethylthiourea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,3-dimethyl-1,3-dipropylthiourea/COSMO_TZVPD/1,3-dimethyl-1,3-dipropylthiourea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,3-dibutyl-1,3-dimethylthiourea/COSMO_TZVPD/1,3-dibutyl-1,3-dimethylthiourea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,3-dimethyl-1,3-dipentylthiourea/COSMO_TZVPD/1,3-dimethyl-1,3-dipentylthiourea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,3-dihexyl-1,3-dimethylthiourea/COSMO_TZVPD/1,3-dihexyl-1,3-dimethylthiourea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,3-diheptyl-1,3-dimethylthiourea/COSMO_TZVPD/1,3-diheptyl-1,3-dimethylthiourea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,3-dimethyl-1,3-dioctylthiourea/COSMO_TZVPD/1,3-dimethyl-1,3-dioctylthiourea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-ethyl-1,3,3-trimethylthiourea/COSMO_TZVPD/1-ethyl-1,3,3-trimethylthiourea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,1,3-trimethyl-3-propylthiourea/COSMO_TZVPD/1,1,3-trimethyl-3-propylthiourea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-butyl-1,3,3-trimethylthiourea/COSMO_TZVPD/1-butyl-1,3,3-trimethylthiourea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1,1,3-trimethyl-3-pentylthiourea/COSMO_TZVPD/1,1,3-trimethyl-3-pentylthiourea_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hba/1-hexyl-1,3,3-trimethylthiourea/COSMO_TZVPD/1-hexyl-1,3,3-trimethylthiourea_c000.orcacosmo']
        )

hbd = (
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Glycine/COSMO_TZVPD/Glycine_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Alanine/COSMO_TZVPD/Alanine_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Proline/COSMO_TZVPD/Proline_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Valine/COSMO_TZVPD/Valine_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Leucine/COSMO_TZVPD/Leucine_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Isoleucine/COSMO_TZVPD/Isoleucine_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Methionine/COSMO_TZVPD/Methionine_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Phenylalanine/COSMO_TZVPD/Phenylalanine_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Tyrosine/COSMO_TZVPD/Tyrosine_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Tryptophan/COSMO_TZVPD/Tryptophan_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Serine/COSMO_TZVPD/Serine_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Threonine/COSMO_TZVPD/Threonine_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Cysteine/COSMO_TZVPD/Cysteine_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Asparagine/COSMO_TZVPD/Asparagine_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Glutamine/COSMO_TZVPD/Glutamine_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Lysine/COSMO_TZVPD/Lysine_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Histidine/COSMO_TZVPD/Histidine_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Arginine/COSMO_TZVPD/Arginine_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Aspartic_acid/COSMO_TZVPD/Aspartic_acid_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Glutamic_acid/COSMO_TZVPD/Glutamic_acid_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Glyceraldehyde/COSMO_TZVPD/Glyceraldehyde_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Erythrose/COSMO_TZVPD/Erythrose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Threose/COSMO_TZVPD/Threose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Ribose/COSMO_TZVPD/Ribose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Arabinose/COSMO_TZVPD/Arabinose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Xylose/COSMO_TZVPD/Xylose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Lyxose/COSMO_TZVPD/Lyxose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Allose/COSMO_TZVPD/Allose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Altrose/COSMO_TZVPD/Altrose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Glucose/COSMO_TZVPD/Glucose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Mannose/COSMO_TZVPD/Mannose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Gulose/COSMO_TZVPD/Gulose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Idose/COSMO_TZVPD/Idose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Galactose/COSMO_TZVPD/Galactose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Talose/COSMO_TZVPD/Talose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Dihydroxyacetone/COSMO_TZVPD/Dihydroxyacetone_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Erythrulose/COSMO_TZVPD/Erythrulose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Ribulose/COSMO_TZVPD/Ribulose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Xylulose/COSMO_TZVPD/Xylulose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Psicose/COSMO_TZVPD/Psicose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Fructose/COSMO_TZVPD/Fructose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Sorbose/COSMO_TZVPD/Sorbose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Tagatose/COSMO_TZVPD/Tagatose_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Lauric_acid/COSMO_TZVPD/Lauric_acid_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Myristic_acid/COSMO_TZVPD/Myristic_acid_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Palmitic_acid/COSMO_TZVPD/Palmitic_acid_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Stearic_acid/COSMO_TZVPD/Stearic_acid_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Arachidic_acid/COSMO_TZVPD/Arachidic_acid_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Lignoceric_acid/COSMO_TZVPD/Lignoceric_acid_c000.orcacosmo'],
        ['D:/Cheminformatics/act_coef/cosmo-rs/hbd/Cholesterol/COSMO_TZVPD/Cholesterol_c000.orcacosmo']
        )

### define the temperature (Kelvin)
T = 298.15

### create an empty list to store the results
hba_all_results = []
hbd_all_results = []

### loop through all components in hba and hbd tuples
for hba_component in hba:
    for hbd_component in hbd:
        ## define the mixture
        crs.add_molecule(hba_component)
        crs.add_molecule(hbd_component)

        ## define ratios of the mixture
        ratios = [
            np.array([0.0, 1.0]),
            np.array([0.1, 0.9]),
            np.array([0.2, 0.8]),
            np.array([0.3, 0.7]),
            np.array([0.4, 0.6]),
            np.array([0.5, 0.5]),
            np.array([0.6, 0.4]),
            np.array([0.7, 0.3]),
            np.array([0.8, 0.2]),
            np.array([0.9, 0.1]),
            np.array([1.0, 0.0]),
        ]

        hba_results = []  # list to store the results for this hba
        hbd_results = []  # list to store the results for this hba

        for ratio in ratios:
            ## add job
            crs.add_job(x=ratio, T=T, refst='pure_component')

        ## compute COSMO-RS calculation
        results = crs.calculate()
        hba_results.append(results['tot']['lng'][1:, 0])  # store the results for this hba
        hbd_results.append(results['tot']['lng'][:10, 1])  # store the results for this hbd
        hba_all_results.append(hba_results)  # store all hba results
        hbd_all_results.append(hbd_results)  # store all hbd results

        ## clear jobs and clear molecules
        crs.clear_jobs()
        crs.clear_molecules()

### save the results as numpy objects
np.save('hba_gamma.npy', hba_all_results)
np.save('hbd_gamma.npy', hbd_all_results)

### reload the results to confirm
hba_gamma = np.load('hba_gamma.npy', allow_pickle=True)
hbd_gamma = np.load('hbd_gamma.npy', allow_pickle=True)