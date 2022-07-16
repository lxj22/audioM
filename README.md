# 7404 group project

## Digit and Gender Classification on voice signals and LRP(layerwiser relevance propagation). All the work is based on the paper: the Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals (https://arxiv.org/abs/1807.03418).

#### please install all the packages in the requirement.txt

### Dataset
The dataset consists of 30000 audio samples of spoken digits (0-9) of 60 different speakers. The raw aduio data can be donloaded in(https://github.com/soerenab/AudioMNIST/tree/master/data). The data will be stored in folder "data"

There is one directory per speaker holding the audio recordings. 

Additionally "audioMNIST_meta.txt" provides meta information such as gender or age of each speaker. It is needed when processing the preprocess_data.py. 

---

### 1.preprocessing 

The data after transformation will be splited into different file folders based on its classification task such as "digit" or "gender". The processed data will be stored in the folder "preprocessed_data" and in hdf5 format. The name of the file is "AlexNet_{speaker}_{digit}_{sample}.hdf5"). The shape of processed data is [1, 1, 227, 227]. The processed data is a spectrogram and will be used in the AlexNet model.

![spectrogram](/pics/spectrogram.png)

---
### 2.training
#### Models
*AlexNet* will be used in the classification task. The model is stored in the folder "models". 

Please run file **"train.py"** in the "model" folder to train the model. Model will be saved in saved_model (model with the highest accuracy on validation dataset, every 10 epochs and final epoch). Log file will be stored in logs. If you have any pretrained model, please put it in the folder "pretrained_models" and its sub-folder "digit"(for digit task), "gender"(for gender task).

Some arguments:
    parser.add_argument("--epochs",type=int,default=150,help="training epochs")
    
    parser.add_argument("--lr",type=float,default=0.001,help="learning rate")
    
    parser.add_argument("--samples",type=int,default=2048,help="samples being used from training data")
    
    parser.add_argument("--batch_size",type=int,default=256,help="batch size")
    
    parser.add_argument("--selected_num",type=int,default=0,help="which of the testing sample to use, from 0-4 ")
    
    parser.add_argument("--pre_trained",type=str,default=None,help="pre_trained_model")
    
    parser.add_argument("--log_file_name",type=str,default="training_log.log",help="log file name")
    
    parser.add_argument("--task",type=str,default="digit",help="digit or gender")
    
Run file train such as "python train.py --log_file_name="digit_task_log.log" task="digit" "


### 3.testing
Run file **test.py** in the folder "models" to test the pretrained models on testset.

Some arguments:

    parser.add_argument("--samples",type=int,default=2048,help="samples being used from testing data")
    
    parser.add_argument("--batch_size",type=int,default=256,help="batch size")
    
    parser.add_argument("--selected_num",type=int,default=0,help="which of the testing sample to use, from 0-4 ")  
    
    parser.add_argument("--model_name",type=str,default="final_epoch.pth",help="the pre-trained model you want to use ")       
    
    parser.add_argument("--task",type=str,default="digit",help="digit or gender")  
    


### 4. LRP(layerwiser relevance propagation)

Please run file **"lrp.ipynb"** to see the relevance map of spectrogram.

### 5. Visualiztion of log file

Please run file "data_visualization.py" to change the log file into line graph.

![line](/pics/line.jpg)

### reference:

https://github.com/deepfindr/xai-series

https://github.com/soerenab/AudioMNIST
@ARTICLE{becker2018interpreting,
  author    = {Becker, S\"oren and Ackermann, Marcel and Lapuschkin, Sebastian and M\"uller, Klaus-Robert and Samek, Wojciech},
  title     = {Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals},
  journal   = {CoRR},
  volume    = {abs/1807.03418},
  year      = {2018},
  archivePrefix = {arXiv},
  eprint    = {1807.03418},
}




