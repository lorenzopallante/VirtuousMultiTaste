# VirtuousMultiTaste
Official repo of the Multi Taste Predictor developed in the framework of the EU-funded VIRTUOUS project

[![Virtuous button][Virtuous_image]][Virtuous link]

[Virtuous_image]: https://virtuoush2020.com/wp-content/uploads/2021/02/V_logo_h.png
[Virtuous link]: https://virtuoush2020.com/

The VirtuousMultiTaste is also implemented into a webserver interface at XXXXXX

### Repo Structure
The repository is organized in the following folders:

- VirtuousMultiTaste/
>Collecting python codes and sources files to run the umami prediction

- data/
> Collecting the training and the test sets of the model, the prioritized list of molecular descriptors and the external DBs with their relative umami predictions

- samples/
> Including examples files to test the code


### Authors
1. [Lampros Androutsos](https://github.com/lamprosandroutsos)
2. [Lorenzo Pallante](https://github.com/lorenzopallante)
3. [Aigli Korfiati](https://github.com/aiglikorfiati)
4. ....


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

XXXXX PLACEHOLDER XXXXX


------------------
## Acknowledgement
------------------

The present work has been developed as part of the VIRTUOUS project, funded by the European Union’s Horizon 2020 research and innovation program under the Marie Sklodowska-Curie-RISE Grant Agreement No 872181 (https://www.virtuoush2020.com/).
