# MACWAE
Multimodal Arbitrary Conditioning with Wasserstein Autoencoder

## Data
Before running the code, ensure that a ./data directory exists in the project folder:
```console
mkdir -p ./data
```console

Next, acquire the necessary datasets and place them inside the ./data directory. The required datasets are:
- PolyMNIST
- MHD
- CUB-Captions
- CelebA

Make sure the datasets are correctly formatted and accessible before proceeding with training or evaluation.

## Environment Setup
To set up a Conda environment, use the provided environment.yml file:
```console
conda env create -f environment.yml
conda activate PyMacwae
```
*Note:* This environment is configured for CPU-only execution. If you plan to run experiments on a GPU (recommended), you may need to install PyTorch and related libraries compatible with your NVIDIA driver version. Refer to the official PyTorch installation guide (`https://pytorch.org/get-started/locally/#mac-installation`) to find the appropriate versions.

## Run
The experiments can be launched by executing the respective `.sh` script in the project directory. Before running a script, ensure it has executable permissions:
```
chmod +x experiment_run.sh
```
These scripts handle experiment execution with predefined settings. Modify them as needed to customize your runs.

## Experiment Configuration
To modify experiment parameters such as the random seed, number of replicates, and device selection (GPU or CPU), edit the corresponding `.sh` script in the project directory. These scripts define the execution settings for different experiments, allowing users to customize runs according to their needs.

## Hyperparameter Configuration
To adjust training settings or model hyperparameters, edit the `./config/hyperparameters.yml` file. This file defines key parameters such as learning rate, batch size, number of epochs, and architecture-specific settings. Modify these values as needed to optimize performance for your specific use case.