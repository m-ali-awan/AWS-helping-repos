from __future__ import print_function
import argparse
import sys
import os
import json

import pandas as pd

# pytorch

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from model import SimpleNet

def model_fn(model_dir):
    print('Loading model..')
    
    model_info={}
    model_info_path=os.path.join(data_dir,'model_info.pth')
    with open(model_info_path,'rb') as f:
        model_info=torch.load(f)
        
        
    print('model_info: {}'.format(model_info))
    
    # Determine the device and construct the model
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=SimpleNet(model_info['input_dim'],
                   model_info['hidden_dim'],
                   model_info['output_dim'])
    
    # Load the stored model parameters
    
    model_path=os.path.join(data_dir,'model.pth')
    with open(model_path,'rb') as f:
        model.load_state_dict(torch.load(f))
        
    return model.to(device)

def _get_train_loader(batch_size,data_dir):
    print('Get data Loader')
    
    train_data=pd.read_csv(os.path.join(data_dir,'train.csv'),header=None)
    
    train_y=torch.from_numpy(train_data[[0]].values).float().squeeze()
    
    train_x=torch.from_numpy(train_data.drop([0],axis=1).values).float()
    
    # Creating tensor dataset
    
    train_ds=torch.utils.data.TensorDataset(train_x,train_y)
    
    return torch.utils.data.DataLoader(train_ds,batch_size=batch_size)

def train(model,train_loader,epochs,optimizer,criterion,device):
    
    for epoch in range(1,epochs+1):
        model.train()
        total_loss=0
        
        for batch_idx,(data,target) in enumerate(train_loader,1):
            data,target=data.to(device),target.to(device)
            optimizer.zero_grad()
            
            output=model(data)
            loss=criterion(output,target)
            loss.backward()
            optimizer.step()
            
            total_loss+=loss.item()
            
        # Printing epoch stats
        print('Epoch: {}, Loss: {}'.format(epoch,total_loss/len(train_loader)))
        
    save_model(model,args.model_dir)
    
def save_model(model,data_dir):
    print('Saving the model..')
    
    path=os.path.join(data_dir,'model.pth')
    torch.save(model.cpu().state_dict(),path)
    
def save_model_params(model,data_dir):
    print('Saving model parameters')
    
    model_info_path=os.path.join(data_dir,'model_info.pth')
    
    with open(model_info_path,'wb') as f:
        model_info={
            'input_dim':args.input_dim,
            'hidden_dim':args.hidden_dim,
            'output_dim':args.output_dim
        }
        
        torch.save(model_info,f)
        
        
        
if __name__=='__main__':
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument('--hosts',type=list,default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host',type=str,default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir',type=str,default=os.environ['SM_MODEL_DIR'])#  ****** very imortant here  model artifacts are saved,so in save ...fn this is to be written
    parser.add_argument('--data-dir',type=str,default=os.environ['SM_CHANNEL_TRAIN'])
    
    
    parser.add_argument('--batch-size',type=int,default=64,metavar='N',
                       help='input batch size for training default:64')
    
    parser.add_argument('--epochs',type=int,default=10,metavar='N',
                       help='number of epochs to train')
    
    parser.add_argument('--lr',type=float,default=0.001,metavar='N',
                       help='learning rate (default:0.001)')
    parser.add_argument('--seed',type=int,default=1,metavar='S',
                       help='random seed (default:1)')
    
    
    parser.add_argument('--input_dim', type=int, default=3, metavar='IN',
                        help='number of input features to model (default: 2)')
    parser.add_argument('--hidden_dim', type=int, default=10, metavar='H',
                        help='hidden dim of model (default: 10)')
    parser.add_argument('--output_dim', type=int, default=1, metavar='OUT',
                        help='output dim of model (default: 1)')
        
    args=parser.parse_args()
    
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    train_loader=_get_train_loader(args.batch_size,args.data_dir)
    
    model=SimpleNet(args.input_dim,args.hidden_dim,args.output_dim).to(device)
    
    save_model_params(model,args.model_dir)
    
    optimizer=optim.Adam(model.parameters(),lr=args.lr)
    criterion=nn.L1Loss()
    
    train(model,train_loader,args.epochs,optimizer,criterion,device)
       
    
    
        