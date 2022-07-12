from torch.utils.data import DataLoader
from dataset import MyDataset
import torch
import torch.nn as nn
from Alexnet import AudioAlexNet_digit
from Alexnet import AudioAlexNet_gender

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import argparse
import logging
from torch.nn import BCELoss


cuda = torch.device('cuda:0')
root_dir = '../'
splits_path = 'preprocessed_data'
saved_path = os.path.join(root_dir,"saved_model")
log_path = os.path.join(root_dir,"logs")
pre_trained_path = os.path.join(root_dir,"pre_trained_model")

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def main(epochs,lr,samples,batch_size,selected_num,pre_trained,log_file_name,task):
    log_file = os.path.join(log_path,log_file_name)
    logger = get_logger(log_file)
    mytraindata = MyDataset(root_dir,splits_path, selected_num, "train",samples=samples,task=task)
    myvalidatedata = MyDataset(root_dir,splits_path,selected_num, "validate",samples=samples,task=task)
    if pre_trained:
        print("loading pre_trained model: ",pre_trained)
        pre_trained_model_path = os.path.join(pre_trained_path,task,pre_trained)
        mynn=torch.load(pre_trained_model_path)
    else:
        if task == "digit":
            mynn = AudioAlexNet_digit().to(cuda)
        else:
            mynn = AudioAlexNet_gender().to(cuda)
            
    if task == "digit":
        loss = CrossEntropyLoss().to(cuda)
    else:
        loss = BCELoss().to(cuda)
    optim = Adam(mynn.parameters(),lr=lr)
    train_loader = DataLoader(mytraindata, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    validate_loader = DataLoader(myvalidatedata, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    total_len = len(myvalidatedata)
    best_acc = 0.5
    for epoch in range(epochs):
        print("current epoch: ",epoch)
        running_loss= 0
        validation_loss = 0 
        total_correct = 0
        print("training...")
        for data in train_loader:
            optim.zero_grad()
            imgs, targets = data
            imgs, targets = imgs.to(cuda),targets.to(cuda)
            output = mynn(imgs)
            if task=="gender":
                targets= targets.reshape(-1,1)
                targets = targets.to(torch.float32)
            result = loss(output,targets)
            result.backward()
            optim.step()
            running_loss +=result
        print("validating...")
        with torch.no_grad():
            for data in validate_loader:
                imgs,targets = data
                imgs, targets = imgs.to(cuda),targets.to(cuda)
                output = mynn(imgs)
                if task=="gender":
                    targets = targets.reshape(-1,1)
                    targets = targets.to(torch.float32)
                result = loss(output,targets)
                validation_loss += result
                if task=="gender":
                    correct_num = sum(targets.eq(output.round()))
                else:    
                    correct_num = sum(targets.eq(output.argmax(dim=1)))
                total_correct+=correct_num
        
        accuracy =  total_correct/total_len   
        logger.info('Epoch:[{}/{}]\t validation_loss={:.5f}\t training_loss = {:.5f}\t acc={:.5f}'.format(epoch , epochs, validation_loss.item(),running_loss.item(), accuracy.item() ))
        if accuracy> best_acc:
            best_acc = accuracy
            best_name = "best_acc.pth"
            best_name_path = os.path.join(saved_path,best_name)
            torch.save(mynn,best_name_path)
            print("best accuracy model saved in epoch: ",epoch)            
        
        if (epoch%10) == 0:
            filename = "model_epoch"+str(epoch)+".pth"
            filename = os.path.join(saved_path,filename)
            torch.save(mynn,filename)
            print("model saved in epoch: ",epoch)
        else:
            pass
    final_path =  os.path.join(saved_path,"final_epoch.pth")
    torch.save(mynn, final_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("training")
    parser.add_argument("--epochs",type=int,default=150,help="training epochs")
    parser.add_argument("--lr",type=float,default=0.001,help="learning rate")
    parser.add_argument("--samples",type=int,default=2048,help="samples being used from training data")
    parser.add_argument("--batch_size",type=int,default=256,help="batch size")
    parser.add_argument("--selected_num",type=int,default=0,help="which of the testing sample to use, from 0-4 ")
    parser.add_argument("--pre_trained",type=str,default=None,help="pre_trained_model")
    parser.add_argument("--log_file_name",type=str,default="training_log.log",help="log file name")
    parser.add_argument("--task",type=str,default="digit",help="digit or gender")

    args = parser.parse_args()
    
    epochs = args.epochs
    lr = args.lr
    samples = args.samples
    batch_size = args.batch_size
    selected_num = args.selected_num
    pre_trained = args.pre_trained
    log_file_name = args.log_file_name
    task = args.task
    print("start training...")
    
    main(epochs,lr,samples,batch_size,selected_num,pre_trained,log_file_name,task)
    
    print("training is over...")
    

    

    

    
