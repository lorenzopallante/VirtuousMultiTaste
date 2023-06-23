"""
VIRTUOUS MULTI TASTE

The VirtuousMultiTaste tool predict the other/sweet/bitter/umami tastes of query molecules based on their molecular structures.

This tool is mainly based on:
    1. VirtuousMultiTaste.py: a main script which calls the following functionalities
    2. Virtuous.py: library of preprocessing functionalities
    3. testing_fourtaste.py: prediction code

To learn how to run, just type:

    python VirtuousMultiTaste.py --help

usage: VirtuousMultiTaste.py [-h] [-c COMPOUND] [-f FILE] [-v VERBOSE] [-d DIRECTORY] [(-t TYPE)]

VirtuousMultiTaste: ML-based tool to predict multiple taste sensqtions including sweet, bitter, umami, from other tastes

optional arguments:
  -h, --help            show this help message and exit
  -c COMPOUND, --compound COMPOUND
                        query compound (allowed file types are SMILES, FASTA, Inchi, PDB, Sequence, Smarts, pubchem name)
  -f FILE, --file FILE  text file containing SMILES of the query molecules
  -t TYPE, --type TYPE  type of the input file (SMILES, FASTA, Inchi, PDB, Sequence, Smarts, pubchem name). If not specified, an automatic recognition of the input format will be tried
  -d DIRECTORY, --directory DIRECTORY
                        name of the output directory
  -v VERBOSE, --verbose VERBOSE
                        Set verbose mode (default: False; if True print messagges)

To test the code you can submit an example txt file in the examples fodler (test.txt)

The code will create a log file and an output folder containing:

    1. "best_descriptors.csv": a csv file collecting the 15 best molecular descriptors for each processed smiles on which the prediction relies
    2. "descriptors.csv": a csv file collecting all the calculated molecular descriptors for each processed smiles
    3. "result_labels.txt": a txt file containing the predicted taste classes for each processed molecule
    4. "result_dominant_labels.txt": a txt file containing the predicted dominant taste classes for each processed molecule
    5. "predictions.csv": a csv summarising the results of the prediction

---------------
Acknowledgement

The present work has been developed as part of the VIRTUOUS project, funded by the European Union’s Horizon 2020 research and innovation program under the Marie Sklodowska-Curie-RISE Grant Agreement No 872181 (https://www.virtuoush2020.com/).

----------------
Version history:
- Version 0.1.0  - 14/10/2022
- Version 0.1.1  - 23/06/2023: Removing AD and correcting minor issues

"""

__version__ = '0.1.1'
__author__ = 'Virtuous Consortium'

# Import libraries
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.info")
import os
import argparse
import time
import sys
from joblib import Parallel, delayed

# # Import Virtuous Library
import Virtuous

# Import testing_umami.py
import testing_fourtaste

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
logging.basicConfig(level=logging.WARNING)

# define functions for preprocessing
def read_mol(cpnd):
    return Virtuous.ReadMol(cpnd, verbose=args.verbose, type=args.type)

def standardize(m):
    return Virtuous.Standardize(m)

# define the function that takes in a SMILES string and returns its descriptors
def calc_desc(smi):
    desc = Virtuous.CalcDesc(smi, Mordred=True, RDKit=False, pybel=False)
    return desc

if __name__ == "__main__":

    # --- Parsing Input ---
    parser = argparse.ArgumentParser(description='VirtuousMultiTaste: ML-based tool to predict the umami taste')
    parser.add_argument('-c','--compound',help="query compound (allowed file types are SMILES, FASTA, Inchi, PDB, Sequence, Smarts, pubchem name)",default=None)
    parser.add_argument('-f','--file',help="text file containing the query molecules",default=None)
    parser.add_argument('-t', '--type', help="type of the input file (SMILES, FASTA, Inchi, PDB, Sequence, Smarts, pubchem name). If not specified, an automatic recognition of the input format will be tried", default=None)
    parser.add_argument('-d','--directory',help="name of the output directory",default=None)
    parser.add_argument('-v','--verbose',help="Set verbose mode", default=False, action='store_true')
    args = parser.parse_args()

    # --- Print start message
    if args.verbose:
        print ("\n\t   *** VirtuousMultiTaste ***\nAn ML-based algorithm to predict sweet, bitter, umami tastes\n")

    # --- Setting Folders and files ---
    # Stting files needed by the code
    code_path = os.path.realpath(__file__)
    root_dir_path = os.path.dirname(os.path.dirname(code_path))
    src_path = os.path.dirname(code_path) + os.sep + "src" + os.sep
    AD_file = src_path  + "fourtaste_AD_train.pkl"
    maximums_filename1 = src_path  + 'maximums_neworder.txt'
    minimums_filename1 = src_path  + 'minimums_neworder.txt'
    features_filename1 = src_path + 'features_list.txt'
    best_features = src_path + 'fourtaste_best_features.txt'
    missing_imputation_method1 = 2
    normalization_method1 = 1
    model_filename1 = src_path  + 'model2_fourtaste.zip'
    selection_flag1 = 0
    data_been_preprocessed_flag1 = 0
    has_features_header1 = 1
    has_samples_header1 = 1
    training_labels_filename1 = src_path  + 'training_labels.txt'
    length_of_features_from_training_filename1 = src_path  + 'length_of_features_from_training.txt'
    tstamp = time.strftime('%Y_%m_%d_%H_%M')
    selected_comorbidities_string1 = ""
    num_cores = 6 
    # Setting output folders and files
    if args.directory:
        output_folder1 = os.getcwd() + os.sep + args.directory + os.sep
    else:
        output_folder1 = os.getcwd() + os.sep + 'Output_folder_' + str(tstamp) + os.sep
    if not os.path.exists(output_folder1):
        os.makedirs(output_folder1)

        testing_fourtaste.initLogging()
    
    #####################################
    # --- Preprocessing (Virtuous.py) ---

    # 1.1 Defining the SMILES to be processed

    # if user defined only one compound with the --compound directive
    if args.compound:
        query_cpnd = []
        query_cpnd.append(args.compound)

    # if the user defined a txt file collecting multiple molecules
    elif args.file:
        with open(args.file) as f:
            query_cpnd = f.read().splitlines()

    else:
        sys.exit("\n***ERROR!***\nPlease provide a molecule or a txt file containing a list of molecules in the allowed formats!\nUse python ../VirtuousMultiTaste.py --help for further information\n")

    # 1.2 Import compound as a molecule object

    ## serial version
    # mol = [Virtuous.ReadMol(cpnd, verbose=args.verbose) for cpnd in query_cpnd]
    ##

    mol = Parallel(n_jobs=num_cores)(
                delayed(read_mol)(cpnd) for cpnd in query_cpnd)
    
    # 1.3 Standardise molecule with the ChEMBL structure pipeline (https://github.com/chembl/ChEMBL_Structure_Pipeline)

    ## previous version
    # standard = [Virtuous.Standardize(m) for m in mol]
    ##

    standard = Parallel(n_jobs=num_cores)(
            delayed(standardize)(m) for m in mol)
    
    # take only the parent smiles
    issues     = [i[0] for i in standard]
    std_smi    = [i[1] for i in standard]
    parent_smi = [i[2] for i in standard]

    # # 1.4 Check the Applicability Domain (AD)
    # fourtaste_AD_threshold = 0.03
    # ## previous version
    # # check_AD = [Virtuous.TestAD(smi, filename=AD_file, verbose = False, sim_threshold=fourtaste_AD_threshold, neighbors = 5, metric = "tanimoto") for smi in parent_smi]
    # ##
    # check_AD = Parallel(n_jobs=num_cores)(
    #         delayed(test_ad)(smi) for smi in parent_smi)
    # test       = [i[0] for i in check_AD]
    # score      = [i[1] for i in check_AD]
    # sim_smiles = [i[2] for i in check_AD]

    # 1.5 Featurization: Calculation of the molecular descriptors

    ## previous version
    # descs = [Virtuous.CalcDesc(smi, Mordred=True, RDKit=False, pybel=False) for smi in parent_smi]
    # DescValues = []
    # for d in descs:
    #     DescValues.append(d[1])
    # DescNames = descs[0][0]
    ##

    DescNames = []
    DescValues = []
    results = Parallel(n_jobs=num_cores)(
             delayed(calc_desc)(smi) for smi in parent_smi)

    for desc in results:
        DescNames = desc[0]
        DescValues.append(desc[1])

    # save all the descriptors with SMILES in the file "descriptors.csv"
    df = pd.DataFrame(data = DescValues, columns=DescNames)
    df.insert(loc=0, column='SMILES', value=parent_smi)
    df.to_csv(output_folder1 + "descriptors.csv", index=False)

    # save only the best features on which the model relies
    col = np.loadtxt(best_features, dtype="str")
    col = np.insert(col, 0, "SMILES")
    df_best = df[col]
    df_best.to_csv(output_folder1 + "best_descriptors.csv", index=False)

    ##########################################
    # --- Run the model (testing_umami.py) ---
    testset_filename1 = output_folder1 + "descriptors.csv"

    delim   = testing_fourtaste.find_delimiter(testset_filename1)
    dataset = testing_fourtaste.preprocess_specific(testset_filename1, delim, output_folder1, src_path, features_filename1)
    ret     = testing_fourtaste.run_all(dataset, maximums_filename1, minimums_filename1,
                  features_filename1, missing_imputation_method1, normalization_method1,
                  model_filename1, selection_flag1, data_been_preprocessed_flag1, selected_comorbidities_string1,has_features_header1, has_samples_header1, training_labels_filename1, 
                  length_of_features_from_training_filename1, output_folder1)

    testing_fourtaste.logging.info("{}".format(ret[1]))

    # Collect results
    df = pd.read_csv(output_folder1 + "result_labels.txt", sep="\t", header=None)
    # Keep only the prediction values
    df = df[[1, 3, 5, 7]]

    dominant_file = output_folder1 + "result_dominant_label.txt"

    with open(dominant_file, 'r') as f:
        dominants = f.readlines()

    # Rename the dataframe column  to the four tastes 
    df = df.rename(columns={1: 'Bitter', 3: 'Sweet', 5: 'Other', 7: 'Umami'})
    df = df.reindex(columns=['Bitter', 'Sweet', 'Umami', 'Other'])
    df = df.round(2)
    df ["Dominant"] = [d.split()[0].upper() for d in dominants]
    df.insert(loc=0, column='SMILES', value=parent_smi)

    df.to_csv(output_folder1 + "predictions.csv", index=False)

    if args.verbose:
        print("")
        print (df)
        print("")
