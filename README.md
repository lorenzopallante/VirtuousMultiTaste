# VirtuousMultiTaste
Official repo of the Multi Taste Predictor developed in the framework of the EU-funded VIRTUOUS project

[![Virtuous button][Virtuous_image]][Virtuous link]

[Virtuous_image]: https://virtuoush2020.com/wp-content/uploads/2021/02/V_logo_h.png
[Virtuous link]: https://virtuoush2020.com/

The VirtuousMultiTaste is also implemented into a webserver interface at https://virtuous.isi.gr/#/virtuous-multitaste

### Repo Structure
The repository is organized in the following folders:

- VirtuousMultiTaste/
>Collecting python codes and sources files to run the umami prediction

- data/
> Collecting the training and the test sets of the model, the prioritized list of molecular descriptors and the external DBs with their relative umami predictions

- examples/
> Including examples files to test the code

- notebooks/
> Collecting the jupyter notebooks used to test the code


### Authors
1. [Lampros Androutsos](https://github.com/lamprosandroutsos)
2. [Lorenzo Pallante](https://github.com/lorenzopallante)
3. [Aigli Korfiati](https://github.com/aiglikorfiati)


----------------
## Prerequisites
----------------

1. Create conda environment:

        conda create -n myenv python=3.10
        conda activate myenv

2. Install required packages:

        conda install -c conda-forge rdkit chembl_structure_pipeline
        conda install -c mordred-descriptor mordred
        pip install knnimpute joblib Cython scikit-learn==1.1.1 xmltodict pyenchant

3. Clone the `VirtuousMultiTaste` repository from GitHub

        git clone https://github.com/lorenzopallante/VirtuousMultiTaste

Enjoy!        

--------------------------------
## How to use VirtuousMultiTaste
--------------------------------

The main code is `VirtuousMultiTaste.py` within the VirtuousMultiTaste folder.

To learn how to run, just type:

    python VirtuousMultiTaste.py --help

And this will print the help message of the program:

    usage: VirtuousMultiTaste.py [-h] [-c COMPOUND] [-f FILE] [-t TYPE] [-d DIRECTORY] [-v]

        VirtuousMultiTaste: ML-based tool to predict the umami taste

        options:
        -h, --help            show this help message and exit
        -c COMPOUND, --compound COMPOUND
                                query compound (allowed file types are SMILES, FASTA, Inchi, PDB, Sequence, Smarts, pubchem name)
        -f FILE, --file FILE  text file containing the query molecules
        -t TYPE, --type TYPE  type of the input file (SMILES, FASTA, Inchi, PDB, Sequence, Smarts, pubchem name). If not specified, an automatic
                                recognition of the input format will be tried
        -d DIRECTORY, --directory DIRECTORY
                                name of the output directory
        -v, --verbose         Set verbose mode

To test the code you can submit an example txt file in the "samples" fodler (test.txt) 

The code will create a log file and an output folder containing:

    1. "best_descriptors.csv": a csv file collecting the 15 best molecular descriptors for each processed smiles on which the prediction relies
    2. "descriptors.csv": a csv file collecting all the calculated molecular descriptors for each processed smiles
    3. "result_labels.txt": a txt file containing the predicted taste classes for each processed molecule
    4. "result_dominant_labels.txt": a txt file containing the predicted dominant taste classes for each processed molecule
    5. "predictions.csv": a csv summarising the results of the prediction

------------------
## Acknowledgement
------------------

The present work has been developed as part of the VIRTUOUS project, funded by the European Union’s Horizon 2020 research and innovation program under the Marie Sklodowska-Curie-RISE Grant Agreement No 872181 (https://www.virtuoush2020.com/).
