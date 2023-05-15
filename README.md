
# Replication package for "Revisiting File Context for Source Code Summarization" under review at TSE.
## Step 0 - Dataset building

 The complete data and models as well as prediction files can be found at:

- funcom_python:https://drive.google.com/file/d/1WnO3Ibp-7D4O1iqe-16hAozggdgvMcMO/view?usp=sharing
- javastmt_fc: 



## Step 1 - Training
To ensure no recursive errors or edits, clone this git repository and  place the javastmt_fc and python_fc folders after decompresing the data

Create directory outdir, with 4 subdirectories  **outdir/{models, histories, viz, predictions}**
**Use Requirements.txt to get your python 3.x virtual environment in sync with our setup.** Venv is preferred. Common issues that might arise from updating an existing venv and solutions :
- GPU not recognized: checking the compatibility of your gpu cudnn/cuda or other drivers with the keras and tf versions fixes this.
- Tf unable to allocate tensor: upgrade to tf 2.4

To train the model XXY use the following command :
```
time python3 train.py --model-type=XXY --batch-size=50 --epochs=10 --gpu=0
```
Note: list of model names can be found in model.py file.

## Step 2 - Predictions
Training print screen will display the epoch at which the model converges, that is when the validation accuracy is not increase much or just before it starts to decrease and validation loss goes up. Once epoch is identified run the following script and replace file in this example with the trained model epoch and timestamp.

```
python3 predict.py path_to_model_epoch --gpu=0
```
predicted comments for all models are provided in the predictions folder.

## Step 4 - Metrics
Bleu and USE and METEOR scripts have been provided by the name of bleu.py, meteor.py and use_v_score.py all of them can be run with the similar commands
```
 python3 bleu.py path_to_predict_file --data=path_to_data
```
replace path_to_data with javastmt_fc/output or funcom_python/output
