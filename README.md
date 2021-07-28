# ægen
Autoencoders for genomic data compression, classification, imputation, phasing and simulation.

## Overview
ægen is a meta-autoencoder which allows to customize the shape of the autoencoder and specify the desired latent space distribution. Additionally, it allows to use conditioning and/or denoising modes.

## Dependencies

### Environment setup
Assuming a Python virtual environment is set up, the dependencies can be installed with:
```console
$ pip3 install -r requirements.txt
```
The whole project has been developed with Python 3.6.1 and PyTorch 1.4.0.

### Data

#### *Founders' dataset*
Human dataset – the human dataset is composed of public-use human whole genome sequences collected from real world-wide populations. The three sources are the listed below:
- [The 1000 genomes project](http://www.nature.com/articles/nature15393), reporting genomes of 2504 individuals from
26 populations from all continents.
- [The Human Genome Diversity Project](https://science.sciencemag.org/content/367/6484/eaay5012), adding 929 diverse genomes
from 54 geographically, linguistically, and culturally diverse human population.
- [The Simons Genome Diversity Project](https://www.nature.com/articles/nature18964), providing genomes from 300
individuals from 142 diverse populations.

The dataset was pruned to contain only single-ancestry origin individuals, i.e., individuals whose four grandparents self-reported belonging to the same ancestral group. After pruning, the dataset resulted in 2965 single-ancestry phased human genomes, each containing a maternal and paternal copy. For that reason, each sequence could be expanded into two, doubling the number of sequences to 5930, to which we refer as *founders*.

#### Dataset filesystem
The whole code is mounted on a specific data filesystem tree. First, it is needed to define the enviroment variables with the data paths. Those paths can be defined in the `scripts/ini.sh` script: `$USER_PATH` is the environment variable pointing to the root of this repository, `$IN_PATH` is the path for incoming data (training, validation and test sets), and `$OUT_PATH` is the path for outgoing data (logs). 

The input data filesystem is defined as follows:
```
$IN_PATH
└─ data
    ├─ human
    │   ├─ sample maps -> EUR, EAS, AMR, SAS, AFR, OCE, WAS
    │   ├─ human HapMap genetic map (.gmap)
    │   ├─ reference panel metadata (.tsv)
    │   ├─ chr22
    │   │   ├─ VCF file (.vcf)
    │   │   └─ prepared
    │   │       ├─ train -> HDF5 files with single-ancestry simulated data
    │   │       ├─ valid -> HDF5 files with single-ancestry simulated data
    │   │       └─ test  -> HDF5 files with single-ancestry simulated data
    │   └─ other chr can be added
    └─ other species can be added
```

#### Data augmentation
Two types of data augmentation have been used: (1) **offline** and (2) **online**. Offline data augmentation precomputes the training set data before starting training the model, whereas online simulation simulates new data samples on-the-fly. In this section, steps to perform offline simulation are explained. In order to specify which type of simulation to use, define that in the `params.yaml` file in the root folder of this repository.

*Founders* have been split in three non-overlapping groups with proportions 80%, 10% and 10%, to generate the training, validation and test sets. For each
set, several datasets have been simulated with the corresponding subset of founders using Wright-Fisher simulation within each population separately and basing the recombination on the human HapMap genetic map. 

In order to run the offline simulation, execute the following commands:
```console
$ cd scripts
$ source ini.sh
## Create single-ancestry maps and simulate single-ancestry individuals within each split.
$ ./simulate.sh species=human generations=[desired num of generations] individuals=[desired num of ind/gen]
## Create HDF5 datasets for each split with the specified number of SNPs.
$ ./create.sh snps=[desired num snps]
```

## Training
The proposed method consists of a highly-adaptable and modular autoencoder that accepts flags to switch to conditioning mode, use different encoder/decoder architectures and specifiy the distribution at the bottleneck of the model. Furthermore, the model accepts two sets: (1) **a set of fixed parameters**, which defines the shape of the network, conditioning, number and size of layers in the encoder/decoder, dropouts, batch normalization and activation functions; (2) **a set of hyperparameters**, which defines optimizer flags and values, such as, the learning rate, weight decay, data augmentation simulation mode, among others. All of those parameters and hyperparameters are defined in the `params.yaml` file in the root folder of this repository. Once both sets have been specifies, a training session can be started by using:
```console
$ cd scripts
$ source ini.sh

## Store the params.yaml file used in this experiment in $OUT_PATH
$ rm -rf $OUT_PATH/experiments/exp[number]
$ mkdir -p $OUT_PATH/experiments/exp[number]
$ cp $USER_PATH/params.yaml $OUT_PATH/experiments/exp[number]/
$ touch $OUT_PATH/experiments/exp[number]/exp[number].log
$ chmod +rwx $OUT_PATH/experiments/exp[number]/exp[number].log

$ python3 $USER_PATH/src/trainer.py \
--species human \
--chr 22 \
--params $OUT_PATH/experiments/exp[number]/params.yaml \
--num [number] \
--verbose False \
--evolution False
```
Or, if using a Slurm queue, running `./submit.sh experiment=[number]` in the `scripts` folder.

## Evaluation
TODO

## Pre-trained models
TODO




