# MPDD Baseline Code
The baseline system provided for the MM 2025 MPDD Challenge serves as a starting point for participants to develop their solutions for the Multimodal Personalized Depression Detection tasks. The baseline system is designed to be straightforward yet effective, providing participants with a solid foundation upon which they can build and improve.

# Results
The metrics reported are weighted/unweighted F1-score (W_F1/U_F1) and weighted/unweighted accuracy (W_Acc./U_Acc.) with and without personalized features (PF) for the MPDD-Young and MPDD-Elderly datasets. Each value represents the best-performing feature combination for each experiment, using default hyper-parameters.

#### MPDD-Elderly (Track1)

| Length | Task Type | Audio Feature | Visual Feature | w/ PF (W_F1/U_F1) | w/ PF (W_Acc./U_Acc.) | w/o PF (W_F1/U_F1) | w/o PF (W_Acc./U_Acc.) |
|--------|-----------|---------------|----------------|-------------------|-----------------------|--------------------|------------------------|
| 1s     | Binary    | mfcc          | openface       | 85.71 / 79.13     | 85.40 / 84.62         | 82.60 / 70.89      | 69.37 / 83.33          |
| 1s     | Ternary   | opensmile     | resnet         | 56.48 / 55.64     | 55.49 / 56.41         | 54.35 / 49.14      | 48.93 / 55.13          |
| 1s     | Quinary   | opensmile     | densenet       | 66.26 / 46.66     | 45.79 / 69.23         | 63.85 / 44.00      | 42.45 / 66.67          |
| 5s     | Binary    | opensmile     | resnet         | 81.75 / 72.37     | 75.40 / 80.77         | 77.90 / 66.15      | 67.94 / 76.92          |
| 5s     | Ternary   | wav2vec       | openface       | 58.22 / 59.37     | 59.62 / 57.69         | 50.88 / 47.59      | 46.58 / 50.00          |
| 5s     | Quinary   | mfcc          | densenet       | 75.62 / 58.40     | 57.71 / 78.21         | 73.49 / 56.83      | 56.98 / 75.64          |


#### MPDD-Young (Track2)

| Length | Task Type | Audio Feature | Visual Feature | w/ PF (W_F1/U_F1) | w/ PF (W_Acc./U_Acc.) | w/o PF (W_F1/U_F1) | w/o PF (W_Acc./U_Acc.) |
|--------|-----------|---------------|----------------|-------------------|-----------------------|--------------------|------------------------|
| 1s     | Binary    | wav2vec       | openface       | 59.96 / 59.96     | 63.64 / 63.64         | 55.23 / 55.23      | 56.06 / 56.06          |
| 1s     | Ternary   | mfcc          | densenet       | 51.86 / 51.62     | 49.66 / 51.52         | 47.95 / 43.72      | 42.63 / 48.48          |
| 5s     | Binary    | opensmile     | resnet         | 62.11 / 62.11     | 62.12 / 62.12         | 60.02 / 60.02      | 60.61 / 60.61          |
| 5s     | Ternary   | mfcc          | densenet       | 48.18 / 41.31     | 41.71 / 50.00         | 42.82 / 39.38      | 41.29 / 42.42          |

# Environment

    python 3.10.0
    pytorch 2.3.0
    scikit-learn 1.5.1
    pandas 2.2.2

Given `requirements.txt`, we recommend users to configure their environment via conda with the following steps:

    conda create -n mpdd python=3.10 -y   
    conda activate mpdd  
    pip install -r requirements.txt 

# Features

In our baseline, we use the following features:

### Acoustic Feature:
**Wav2vec：** We extract utterance-level acoustic features using the wav2vec model pre-trained on large-scale audio data. The embedding size of the acoustic features is 512.  
The link of the pre-trained model is: [wav2vec model](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)

**MFCCs：** We extract Mel-frequency cepstral coefficients (MFCCs). The embedding size of MFCCs is 64.  

**OpenSmile：** We extract utterance-level acoustic features using opensmile. The embedding size of OpenSMILE features is 6373.  

### Visual Feature:
**Resnet-50 and Densenet-121：** We employ OpenCV tool to extract scene pictures from each video, capturing frames at a 10-frame interval. Subsequently, we utilize the Resnet-50 and Densenet-121 model to generate utterance-level features for the extracted scene pictures in the videos. The embedding size of the visual features is 1000 for Resnet, and 1024 (Track1) or 1000 (Track2) for Densenet.
The links of the pre-trained models are:  
 [ResNet-50](https://huggingface.co/microsoft/resnet-50)  
 [DenseNet-121](https://huggingface.co/pytorch/vision/v0.10.0/densenet121)  

**OpenFace：** We extract csv visual features using the pretrained OpenFace model. The embedding size of OpenFace features is 709. You can download the executable file and model files for OpenFace from the following link: [OpenFace Toolkit](https://github.com/TadasBaltrusaitis/OpenFace)

### Personalized Feature:
We generate personalized features by loading the GLM3 model, creating personalized descriptions, and embedding these descriptions using the `roberta-large` model. The embedding size of the personalized features is 1024.  
The link of the `roberta-large` model is: [RoBERTa Large](https://huggingface.co/roberta-large)

# Usage
## Dataset Download
Given the potential ethical risks and privacy concerns associated with this dataset, we place the highest priority on the protection and lawful use of the data. To this end, we have established and implemented a series of stringent access and authorization management measures to ensure compliance with relevant laws, regulations, and ethical standards, while making every effort to prevent potential ethical disputes arising from improper data use.  

To further safeguard the security and compliance of the data, please complete the following steps before contacting us to request access to the dataset labels and extracted features:  

- **1. Download the [MPDD Dataset License Agreement PDF](https://github.com/hacilab/MPDD/blob/main/MPDD%20Dataset%20License%20Agreementt.pdf)**.

- **2. Carefully review the agreement**: The agreement outlines in detail the usage specifications, restrictions, and the responsibilities and obligations of the licensee. Please read the document thoroughly to ensure complete understanding of the terms and conditions.  

- **3. Manually sign the agreement**: After confirming your full understanding and agreement with the terms, fill in the required fields and sign the agreement by hand as formal acknowledgment of your acceptance (should be signed with a full-time faculty or researcher).  

Once you have completed the above steps, please submit the required materials to us through the following channels:  

- **Primary contact email**: sstcneu@163.com  
- **CC email**: fuchangzeng@qhd.neu.edu.cn  

We will review your submission to verify that you meet the access requirements. Upon approval, we will grant you the corresponding data access permissions. Please note that all materials submitted will be used solely for identity verification and access management and will not be used for any other purpose.  

We sincerely appreciate your cooperation in protecting data privacy and ensuring compliant use. If you have any questions or require further guidance, please feel free to contact us via the emails provided above.

After obtaining the dataset, users should modify `data_rootpath` in the scripts during training and testing. Notice that testing data will be made public in the later stage of the competition.

`data_rootpath`:

    ├── Training/
    │   ├──1s
    │   ├──5s
    │   ├──individualEmbedding
    │   ├──labels
    ├── Testing/
    │   ├──1s
    │   ├──5s
    │   ├──individualEmbedding
    │   ├──labels


## Training
To train the model with default parameters, taking MPDD-Young for example, simply run:  

```bash
cd path/to/MPDD   # replace with actual path
```
```bash
bash scripts/Track2/train_1s_binary.sh
```

You can also modify parameters such as feature types, split window time, classification dimensions, or learning rate directly through the command line:  
```bash
bash scripts/Track2/train_1s_binary.sh --audiofeature_method=wav2vec --videofeature_method=resnet --splitwindow_time=5s --labelcount=3 --batch_size=32 --lr=0.001 --num_epochs=500
```
Refer to `config.json` for more parameters.

The specific dimensions of each feature are shown in the table below:
| Feature                  | Dimension                       |
|--------------------------|---------------------------------|
| Wav2vec                 | 512                              |
| MFCCs                   | 64                               |
| OpenSmile               | 6373                             |
| ResNet-50               | 1000                             |
| DenseNet-121            | 1024 for Track1, 1000 for Track2 |
| OpenFace                | 709                              |
| Personalized Feature    | 1024                             |


## Testing
To predict the labels for the testing set with your obtained model, first modify the default parameters in `test.sh` to match the current task, and run:  

```bash
cd path/to/MPDD   # replace with actual path
```
```bash
bash scripts/test.sh
```
After testing 6 tasks in Track1 or 4 tasks in Track2, the results will be merged into the `submission.csv` file in `./answer_Track2/`.

# Acknowledgements
The benchmark of MPDD is developed based on the work of MEIJU 2025. The Github URL of MEIJU 2025 is: https://github.com/AI-S2-Lab/MEIJU2025-baseline.
