Graph deep learning on whole slide image predicts the context-aware prognostic pathological features of renal cell carcinoma
=====================

## Dependencies
* To install the dependencies for this project, see the "requirements.yaml"
* Tested on Nvidia TESLA V100 x 2 with CUDA 11.1

## Processing Whole slide image (WSI) into superpatch graph
* First, run the ./Superpatch_network_construction/supernode_generation.py
* And then, run the ./Superpatch_network_construction/superpatch-network_construction.py

## Training TEA-graph using superpatch graph
* Run the main.py with appropriate hyperparameters

## Visualization of IG (Integrated gradients) value on WSI
* Run the Subgraph_visualization.py

## Acknowledgments
* http://github.com/mahmoodlab/Patch-GCN
* http://github.com/lukemelas/EfficientNet-PyTorch
* http://github.com/pyg-team/pytorch_geometric

BiNEL (http://binel.snu.ac.kr) - This code is made available under the MIT License and is available for non-commercial academic purposes