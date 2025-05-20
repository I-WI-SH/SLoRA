# SLoRA: Spiking Low-Rank Adaptation for Power-Efficient Fine-Tuning
In this folder, we provide the implementation code for the experiments described in this paper, which is also accessible via the following link: [SLoRA](https://anonymous.4open.science/r/SLoRA-26A2).

## Requirements
in ./SLoRA
pip install -e.
pip install -r requirements.txt

in ./SLoRA/loralib
pip install -e.

## Datasets
When running the scripts, the GLUE dataset will be downloaded automatically. If an internet connection is unavailable, please download the dataset locally beforehand.

## Models
If using a local model, please modify the model_name_or_path parameter in the scripts.

## Run
in ./SLoRA
run `bash ./scripts/cola.sh` for cola task

## Results
### GLUE benchmark
![alt text](expresult.png)

### AdvGLUE benchmark
![alt text](advresult.png)

## Acknowledgments
Parts of the code are inspired by [AdaLoRA](https://github.com/QingruZhang/AdaLoRA).
