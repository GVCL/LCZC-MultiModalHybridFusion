MultiModal Hybrid Fusion Strategies for Local Climate Zone Classification

 Introduction
   
Rapid urbanization alters land surface characteristics through increased building density, construction materials, transportation infrastructure, and various anthropogenic activities. As a result, urban areas often experience warmer temperatures compared to their surrounding rural regions, a phenomenon known as the Urban Heat Island (UHI) effect. UHI intensifies extreme heat events and can influence local environmental conditions such as precipitation patterns and air pollution levels. 

Understanding the spatial structure of urban environments is therefore important for climate studies, urban planning, and environmental monitoring. Traditional land cover classification systems provide only a limited number of urban categories. To address this limitation, the Ian D. Stewart and Timothy R. Oke (2012) introduced Local Climate Zone (LCZ) framework  which consist of  17 classes;   10 built up class (LCZ 1–10) and 7 natural class (LCZ A–G). 

In this work, we develop and implement deep learning-based multimodal fusion strategies to integrate SAR and multispectral data for LCZ classification. The models are based on convolutional neural network (CNN) architectures and employ a hybrid fusion strategy that combines data- and feature-level fusion.  

Overview of the repository:

This repository includes implementations of multiple fusion models. The models for  different fusion strategies of SAR and MSI data from Sentinel-1 (S1) and Sentinel-2 (S2), respectively, for LCZ classification, that extend a baseline hybrid fusion architecture with additional mechanisms including attention mechanism, multiscale Gaussian smoothing, and late fusion.

(i)   FM1-model.py- A baseline hybrid fusion model for pixel- and feature-level integration 

(ii)  FM2-model.py - An attention-based hybrid fusion model

(iii) FM3-model.py - A multi-scale hybrid fusion model

(iv)	FM4-model.py - A decision-level fusion model, which uses a weighted combination of CNN and U-Net outputs of SAR and MSI modalities, respectively. 

Data:

The models are trained and evaluated using the So2Sat LCZ42 data. This dataset can be accessed from http://doi.org/10.14459/2018mp1483140. 

This dataset provides:

- Sentinel-1 SAR imagery

- Sentinel-2 multispectral imagery

- Local Climate Zone labels 

Requirements:

Dependencies for running the models include:

    Python 3.10.13
   
    TensorFlow  
   
    Keras
   
    NumPy
   
    SciPy
   
    GDAL
   
    scikit-learn
   
    matplotlib

Install Dependencies:

       pip install tensorflow keras  numpy scipy scikit-learn matplotlib

How to use the repository:

1.  Clone the repository
   
       git clone https://github.com/GVCL/LCZC-MultiModalHybridFusion.git

      cd HyLCZC-MultiModalHybridFusion

      ls 

      FM1-model.py

      FM2-model.py

      FM3-model.py

      FM4-model.py

3. Prepare the Dataset
   
   Download the publicly available So2Sat LCZ42 dataset and provide the data path in the ‘fileinp’ of the model you want to run (say, FM1-model.py).
   
5. Run the fusion model
   
    To run  any fusion model
   
python FM1-model.py

python FM1-model.py

Inorder to run the model for the complete dataset of 352366 patches, need a high memory server. This code was executed on a high memory node of configuration high-memory compute node of 2× Intel Xeon Cascade Lake 8268, 24 cores, 2.9 GHz, processors with 768 GB memory. The model is run for a period of 100 epochs.

Batch Scirpt used to run on high memory compute node:

#!/bin/sh

#SBATCH -N 1

#SBATCH --ntasks-per-node=1

#SBATCH --time=72:00:00

#SBATCH --exclusive

#SBATCH --job-name=fm1

#SBATCH --error=fm1.%J

#SBATCH --output=fm1.python.%J

#SBATCH --partition=hm

conda init

conda activate tmp

time python FM1-model.py

For the FM1 model, took 209m18.615s to complete the run.

4. Load Pretrained model weightes
   
   Pretrained models are available for download at:   
https://huggingface.co/datasets/ancythomas/TrainedFusionModel/tree/main

These files contain the trained weights for the different fusion strategies. After downloading the required model file, it can be loaded directly into the framework for inference or evaluation or can be used for transfer learning for the related application.

To load a pretrained model:

from tensorflow.keras.models import load_model

model = load_model("FM1G-100.h5")

ls TrainedFusionModel/

a indicates data-level fusion; b for feature-level fusion;  G indicates Band Grouping 

Baseline Hybrid Fusion Model 

FM1-100.h5 

FM1G-100.h5

FM1a-100.h5

FM1aG-100.h5

FM1b-100.h5

FM1bG-100.h5


Hybrid fusion with Attention mechanism

FM2-100.h5

FM2G-100.h5

FM2b-100.h5

FM2bG-100.h5

Hybrid fusion with Multiscale feature extraction

FM3-100.h5

FM3Gfinal-100.h5

FM3a-100.h5

FM3aGfinal-100.h5

FM3b-100.h5

FM3bGfinal-100.h5

Decision level weighted fusion

FM4-.2-100.h5

FM4G-.2-100.h5

5. Applications
   
The proposed fusion models support research in:

•	Remote sensing data fusion

•	Deep learning for Earth observation

•	Transfer learning 

•	Local Climate Zone analysis


Citation

If you use this repository in your research, please cite the related publication (to be added).  


