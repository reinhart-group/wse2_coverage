# wse2_coverage
This project used regression and segmentation models to obtain the monolayer crystal coverage of WSe2 from their micrographs. 

# Objective
- to explore the behavior of regression models compared to segmentation models for characterizing micrographs, such as determining the thin film crystal growth on a substrate, quantified by crystal coverage
- to investigate how pretraining domains and different modes of transfer learning impact the capabilities and reliability of models at inferencing crystal coverage of samples not included in the training

# Data
Processed data for this project can be found at https://doi.org/10.5281/zenodo.10784189
8432222
Raw data can be found at https://m4-2dcc.vmhost.psu.edu/list/data/RVJkDr8j1RPU

# Workflows
- Regression models: notebooks/regression
- Segmentation models: notebooks/segmentation
- Plots: notebooks/plots
- codes: codes
- plots and data from models: Result
. 
# To cite
@article{moses2023crystal,
  title={Crystal Growth Characterization of WSe $ \_2 $ Thin Film Using Machine Learning},
  author={Moses, Isaiah A and Wu, Chengyin and Reinhart, Wesley F},
  journal={arXiv preprint arXiv:2312.14311},
  year={2023}
}



