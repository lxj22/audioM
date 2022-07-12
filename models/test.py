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
from torch.nn import BCELoss

cuda = torch.device('cuda:0')
root_dir = '../'
splits_path = 'preprocessed_data'
read_path = os.path.join(root_dir,"pre_trained_model")

def main(model_name,samples,batch_size,selected_num,task):
    model_ = os.path.join(read_path,task,model_name)
    model = torch.load(model_)
    mytestdata = MyDataset(root_dir, splits_path, selected_num,"test",samples=samples,task=task)
    test_loader = DataLoader(mytestdata, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    if task == "digit":
        loss = CrossEntropyLoss().to(cuda)
    else:
        loss = BCELoss().to(cuda)
    total_len = len(mytestdata)
    test_loss = 0
    total_correct = 0
    with torch.no_grad():
        for data in test_loader:
            imgs,targets = data
            imgs, targets = imgs.to(cuda),targets.to(cuda)
            output = model(imgs)
            if task=="gender":
                targets = targets.reshape(-1,1)
                targets = targets.to(torch.float32)
            result = loss(output,targets)
            test_loss += result
            if task=="gender":
                correct_num = sum(targets.eq(output.round()))
            else:    
                correct_num = sum(targets.eq(output.argmax(dim=1)))            
            total_correct+=correct_num
            
    accuracy =  total_correct/total_len     
    print("test accuracy: ",accuracy.item())        
    print("test loss: ",test_loss.item())
    
    


        


if __name__ == "__main__":
    parser = argparse.ArgumentParser("testing")
    parser.add_argument("--samples",type=int,default=2048,help="samples being used from testing data")
    parser.add_argument("--batch_size",type=int,default=256,help="batch size")
    parser.add_argument("--selected_num",type=int,default=0,help="which of the testing sample to use, from 0-4 ")  
    parser.add_argument("--model_name",type=str,default="final_epoch.pth",help="the pre-trained model you want to use ")       
    parser.add_argument("--task",type=str,default="digit",help="digit or gender")       
        
    args = parser.parse_args()

    samples = args.samples
    batch_size = args.batch_size
    selected_num = args.selected_num
    model_name =args.model_name
    task = args.task
    print("start testing...")
    
    main(model_name,samples,batch_size,selected_num,task)
    
    print("testing is over...")
        