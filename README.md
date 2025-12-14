
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


# Please cite our paper if you use our code or dataset:

Fu, C., Fu, Z., Zhang, Q., Kuang, X., Dong, J., Su, K., ... & Ishiguro, H. (2025, October). The First MPDD Challenge: Multimodal Personality-aware Depression Detection. In Proceedings of the 33rd ACM International Conference on Multimedia (pp. 13924-13929).


Refer to our [website](https://hacilab.github.io/MPDDChallenge.github.io/) for more information: https://hacilab.github.io/MPDDChallenge.github.io/.
