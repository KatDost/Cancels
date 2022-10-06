# Cancels
To combat a continuous specialization of datasets to what is already known, Cancels aims to identify areas in the chemical compound space that fall short and suggests additional experiments that help bridge the gap

Predicting in advance the behavior of new chemical compounds can support the design process of new products by directing the research towards the most promising candidates and ruling out others.
Such predictive models can be data-driven using Machine Learning or based on researchers' experience and depend on the collection of past results. In either case: models (or researchers) can only make reliable assumptions on compounds that are similar to what they have seen before.
Therefore, consequent usage of these predictive models shapes the dataset and causes a continuous specialization shrinking the applicability domain of all trained models on this dataset in the future, and increasingly harming model-based exploration of the space.

In our paper, we propose Cancels (CounterActiNg Compound spEciaLization biaS), a technique that helps to break the dataset specialization spiral. 
Aiming for a smooth distribution of the compounds in the dataset, we identify areas in the space that fall short and suggest additional experiments that help bridge the gap.
Thereby, we generally improve the dataset quality in an entirely unsupervised manner and create awareness of potential flaws in the data. 
Cancels does not aim to cover the entire compound space and hence retains a desirable degree of specialization to a specified research domain. 

An extensive set of experiments on the use-case of biodegradation pathway prediction not only reveals that the bias spiral can indeed be observed but also that Cancels produces meaningful results.
Additionally, we demonstrate that mitigating the observed bias is crucial as it cannot only intervene with the continuous specialization process, but also significantly improves a predictor's performance while reducing the amount of required experiments. 
Overall, we believe that Cancels can support researchers in their experimentation process to not only better understand their data and potential flaws, but also to grow the dataset in a sustainable way.
All code and results supporting our claims made in the paper are available here.

# Citation
If you want to use this implementation or cite Imitate in your publication, please cite the following [paper](https://doi.org/10.21203/rs.3.rs-2133331/v1):
```
Katharina Dost, Zac Pullar-Strecker, Liam Brydon, Kunyang Zhang, Jasmin Hafner, Patricia Riddle, and Jörg Wicker.
"Combatting Over-Specialization Bias in Growing Chemical Databases."
05 October 2022, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-2133331/v1]
```
<!---
Bibtex:
```
@INPROCEEDINGS {Dost2022,
author = {K. Dost and Z. Pullar-Strecker and L. Brydon and K. Zhang and J. Hafner and P. Riddle and J. Wicker},
title = {Combatting Over-Specialization Bias in Growing Chemical Databases},
year = {2022},
}
```
--->

# How to use Cancels
Cancels is integrated in the PyPI package <a rel="imitatebias" href="https://pypi.org/project/imitatebias/">imitatebias</a> which can be installed with
```
pip install imitatebias
```
Please see the package documentation for usage examples. 

We provide a self-contained implementation of all methods and results presented in the paper here. This implementation is not dependent on the PyPI package. Our key results can be found in the following Jupyter notebooks:
- [Analysis_BBD_SOIL.ipynb](Analysis_BBD_SOIL.ipynb): Analysis of the datasets BBD and SOIL (Fig. 5-9 in the paper)
- [Evaluation_Tox21.ipynb](Evaluation_Tox21.ipynb): Results evaluated on the Tox21 dataset (Fig. 10-14 in the paper)

Datasets and additional files for data handling are provided here:
- [Data/](Data/): Folder containing all datasets and intermediate pre-processing steps
- [Data_Preprocessing.ipynb](Data_Preprocessing.ipynb): Data handling and pre-processing

Note that, due to Github's file size limitations, we had to .zip large files. If you wish to reproduce our results, you may need to unpack those .zip files before running our code.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
