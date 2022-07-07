# Localized-Shape-Modelling-with-Global-Coherence
Code of the paper [*"Localized Shape Modelling with Global Coherence: An Inverse Spectral Approach"*](https://arxiv.org/abs/2108.02161)

![Teaser](media/NewTeaserFull.png)

## Setup

Load the coda environment: 
    `conda env create -f environment.yml`

Download the folders containing the datasets as .mat files from this link:
> [https://1drv.ms/u/s!AkyShb75UDGyarSEgmNmiZCQ5KU?e=ReJMnI](https://drive.google.com/drive/folders/169IYkluW3s-6wfinGQlZ85e2n4AWLRKE?usp=sharing)

## Training

You can train the model on different datasets.

To train the model on SURREAL, run: `python TrainingSURREAL.py`
    
To train the model on CUBE, run: `python TrainingCUBE.py`

## Testing

Once the training is completed, you can test the model with: 
    `python intrinsic_error.py`


## Citation

If you use our work, please cite our papers.

```bibtex
@article {pegoraro2021localized, 
 journal = {Computer Graphics Forum}, 
 title = {{Localized Shape Modelling with Global Coherence: An Inverse Spectral Approach}}, 
 author = {Pegoraro, Marco and Melzi, Simone and Castellani, Umberto and Marin, Riccardo and Rodolà, Emanuele},
 year = {2022}, 
 publisher = {The Eurographics Association and John Wiley & Sons Ltd.}, 
 ISSN = {1467-8659},
 DOI = {10.1111/cgf.14599}
}
```
