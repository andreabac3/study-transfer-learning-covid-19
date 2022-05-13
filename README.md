<div align="center">    
 
# Study on transfer learning capabilities for pneumonia classification in chest-x-rays images


[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This repository contains the official code of the research paper [Study on transfer learning capabilities for pneumonia classification in chest-x-rays images](https://www.sciencedirect.com/science/article/pii/S0169260722002152) pubblished at the [Computer Methods and Programs in Biomedicine](https://www.journals.elsevier.com/computer-methods-and-programs-in-biomedicine) Journal.


</div>


## Cite this work
```bibtex
@article{avola2022study,
  title={Study on Transfer Learning Capabilities for Pneumonia Classification in Chest-X-Rays Images},
  author={Avola, Danilo and Bacciu, Andrea and Cinque, Luigi and Fagioli, Alessio and Marini, Marco Raoul and Taiello, Riccardo},
  journal={Computer Methods and Programs in Biomedicine},
  pages={106833},
  year={2022},
  publisher={Elsevier}
}
```

## How to install
You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:
```sh
git clone https://github.com/
```
Then you should install all the dependencies. 
To do this we suggest to use the setup.sh file to create a new conda environment.
```sh
bash setup.sh
```

## Dataset construction
To be compliant to copyright issue, we release the file list that compose our dataset.
To build the dataset: <br>
(i) You should download the Kaggle data from the [competition page](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). <br>

(ii) For the covid data you can download the images from this [github repository](https://github.com/ieee8023/covid-chestxray-dataset). <br>

(iii) The downloaded data must be re-arranged following the dataset structure present in the file data/mia/dataset/dataset_structure.txt .
```
mia
└── dataset
    ├── test
    │   ├── BACTERIA
    │   │   ├── person100_bacteria_475.jpeg
    │   │   ├── person100_bacteria_477.jpeg
...
```


## To run your experiments
You can use the train_and_test.sh file.
```sh
bash train_and_test.sh
```

# Authors

* **Danilo Avola**  - [website](https://www.di.uniroma1.it/it/docenti/avola-danilo)
* **Andrea Bacciu**  - [github](https://github.com/andreabac3) - [website](https://andreabac3.github.io)
* **Luigi Cinque**  - [website](https://www.di.uniroma1.it/it/docenti/cinque-luigi)
* **Alessio Fagioli**  - [github](https://sites.google.com/uniroma1.it/alessiofagioli-eng/home)
* **Marco Raoul Marini**  - [website](https://www.marcoraoulmarini.it)
* **Riccardo Taiello**  - [github](https://github.com/rtaiello) - [website](https://rtaiello.github.io)