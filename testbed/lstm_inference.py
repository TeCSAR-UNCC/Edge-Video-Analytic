import numpy as np
import os
del os.environ['MKL_NUM_THREADS']
import torch
import torch.nn as nn
import torch.optim as optim
import sys

'''     Device configuration      '''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' HYPERPARAMETERS'''
num_steps_to_predict=5
num_steps = 3
modelep = 110
input_size = 75
hidden_size = 256
num_layers=1
num_classes = 4*num_steps_to_predict
batch_size = 1
num_videos = 24  
original_fps=60
desired_fps=5
jump=int(original_fps/desired_fps)
path_to_model = './lstm_model/model_step'+str(num_steps)+'_'+str(modelep)+'_.ckpt' 

w_img, h_img=1920.0,1080.0

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size,num_classes)

    def forward(self,input_x):

        #Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, input_x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, input_x.size(0), self.hidden_size).to(device)

        #Forward propagate LSTM
        #out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(input_x,(h0,c0))

        out = self.fc(out[:,-1,:])
        return out
        
#class lstm_inference:        
def inference(inputs,num_objects):
    #input_batch=float(sys.argv[1])
    #dummy_input = np.zeros([1, 3,75])
    input_batch=np.reshape(inputs,[num_objects,3,75])
    input_batch=np.asarray(input_batch,dtype=np.float32)
    inputs=input_batch
    #print(input_batch)
    input_batch=(torch.from_numpy(input_batch)).to(device)
    model = RNN(input_size,hidden_size,num_layers,num_classes).to(device)
    model.load_state_dict(torch.load(path_to_model))
    #predictions=model(input_batch)
    #print(predictions)
    output=model(input_batch)
    with torch.no_grad():
        output=output.cpu().numpy()
    return np.reshape(output,[num_objects*20,])    

def check_input(inputs):
    return inputs    
       
#if __name__ == '__main__':

    #sys.stdout.write(str(squared(x)))        





        
