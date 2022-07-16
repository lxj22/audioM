# 7404 group project

## Digit and Gender Classification on voice signals and LRP(layerwiser relevance propagation). All the work is based on the paper: the Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals (https://arxiv.org/abs/1807.03418).

### Dataset
The dataset consists of 30000 audio samples of spoken digits (0-9) of 60 different speakers. The raw aduio data can be donloaded in(https://github.com/soerenab/AudioMNIST/tree/master/data). The data will be stored in folder "data"

There is one directory per speaker holding the audio recordings. 

Additionally "audioMNIST_meta.txt" provides meta information such as gender or age of each speaker. It is needed when processing the preprocess_data.py. The data after transformation will be splited into different file folders based on its classification task such as "digit" or "gender". The processed data will be stored in the folder "preprocessed_data". The shape of processed data is [1, 1, 227, 227]. It is represented by a spectrogram. 





### 1.preprocessing 
### 2.training
### 3.testing

#### please install all the packages in the requirement.txt



