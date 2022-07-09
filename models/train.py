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

def main(epochs,lr,samples,batch_size):
    mytraindata = MyDataset(root_dir, splits_path, 0, "train",samples=samples)
    myvalidatedata =  MyDataset(root_dir, splits_path, 0, "validate",samples=samples)
    mynn = AudioAlexNet().to(cuda)
    loss = CrossEntropyLoss().to(cuda)
    optim = Adam(mynn.parameters(),lr=lr)
    train_loader = DataLoader(mytraindata, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    validate_loader = DataLoader(myvalidatedata, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    total_len = len(myvalidatedata)
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
                result = loss(output,targets)
                validation_loss += result
                correct_num = sum(targets.eq(output.argmax(dim=1)))
                total_correct+=correct_num
        accuracy =  total_correct/total_len     
        print("validate accuracy: ",accuracy.item())        
        print("epoch training loss: ",running_loss.item())
        print("epoch validation loss:",validation_loss.item())
        if (epoch%10) == 0:
            filename = "model_epoch"+str(epoch)+".pth"
            filename = os.path.join(saved_path,filename)
            torch.save(mynn.state_dict(),filename)
            print("model saved in epoch: ",epoch)
        else:
            pass
    final_path =  os.path.join(saved_path,"final_epcoh.pth")
    torch.save(mynn, final_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("training")
    parser.add_argument("--epochs",type=int,default=200,help="training epochs")
    parser.add_argument("--lr",type=float,default=0.001,help="learning rate")
    parser.add_argument("--samples",type=int,default=2048,help="samples being used from training data")
    parser.add_argument("--batch_size",type=int,default=256,help="batch size")
    args = parser.parse_args()
    
    epochs = args.epochs
    lr = args.lr
    samples = args.samples
    batch_size = args.batch_size
    print("start training...")
    main(epochs,lr,samples,batch_size)
    
    print("training is over...")
    

    

    

    
