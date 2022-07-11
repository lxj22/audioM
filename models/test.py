from torch.utils.data import DataLoader
from dataset import MyDataset
import torch
import torch.nn as nn
from Alexnet import AudioAlexNet
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import argparse


cuda = torch.device('cuda:0')
root_dir = '../'
splits_path = 'preprocessed_data'
saved_path = os.path.join(root_dir,"saved_model")

def main(model_name,samples,batch_size,selected_num):
    model_ = os.path.join(saved_path,model_name)
    model = torch.load(model_)
    mytestdata = MyDataset(root_dir, splits_path, selected_num, "test",samples=samples)
    test_loader = DataLoader(mytestdata, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    loss = CrossEntropyLoss().to(cuda)

    total_len = len(mytestdata)
    test_loss = 0
    total_correct = 0
    with torch.no_grad():
        for data in test_loader:
            imgs,targets = data
            imgs, targets = imgs.to(cuda),targets.to(cuda)
            output = model(imgs)
            result = loss(output,targets)
            test_loss += result
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
        
    args = parser.parse_args()

    samples = args.samples
    batch_size = args.batch_size
    selected_num = args.selected_num
    model_name =args.model_name
    print("start testing...")
    
    main(model_name,samples,batch_size,selected_num)
    
    print("testing is over...")
        