Hybrid Fusion Strategies  for Local Climate Zone Classification

This repository contains a folder 'Model'. The folder contains different strategies for the fusion of Synthetic Aperture Radar (SAR) and Multispectral (MS) image data.  The strategies include baseline hybrid fusion model (FM1 model) and  its three enhancements [ (a) integration of attention mechanisms (FM2 model), (b) application of multiscale Gaussian filtering (FM3 model), and (c) a decision-level fusion strategy using a weighted U-Net-CNN architecture (FM4 model)] . The fusion models are implemented on So2SatLCZ42 data and evaluation metrics is evaluated on the classification of Local Climate Zone with 17 Class .  



The final classification output from each fusion model is uploaded here https://huggingface.co/datasets/ancythomas/output/tree/main. This output has the trained model obtained for each fusion strategy trained for a period of 100 epochs on So2SatLCZ42 data . Also, the output for the ablation and band grouping experiments for fusion models FM1, FM2, FM3 and FM4 are available here.

output/

-- FM1 model/

FM1-100.h5 FM1a-100.h5 FM1b-100.h5

FM1G-100.h5 FM1aG-100.h5 FM1bG-100.h5 (band grouping)

-- FM2 model/

FM2-100.h5 FM2b-100.h5

FM2G-100.h5 FM2bG-100.h5 (band grouping)

-- FM3 model/

FM3-100.h5 FM3a-100.h5 FM3b-100.h5

FM3G-100.h5 FM3aG-100.h5 FM3bG-100.h5 (band grouping)

-- FM4 model/

FM4-100.h5

FM4G-100.h5 (band grouping)
