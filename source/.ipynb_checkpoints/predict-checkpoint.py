import argparse
import os
import sys
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from io import StringIO
from six import BytesIO

from model import SimpleNet

CONTENT_TYPE='application/x-npy'

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet(model_info['input_dim'], model_info['hidden_dim'], model_info['output_dim'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # prep for testing
    model.to(device).eval()

    print("Done loading model.")
    return model

def input_fn(serialized_input_data,content_type):
    print('Deserializing input data..')
    
    if content_type==CONTENT_TYPE:
        streams=BytesIO(serialized_input_data)
        return np.load(streams)
    raise Exception('Requested unsupported content type in content type '+ content_type)
    

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    if accept == CONTENT_TYPE:
        buffer = BytesIO()
        np.save(buffer, prediction_output)
        return buffer.getvalue(), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)
    

def predict_fn(input_data,model):
    print('Predicting the values....')
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data=torch.from_numpy(input_data.astype('float32'))
    data=data.to(device)
    
    model.eval()
    
    out=model(data)
    
    result=out.cpu().detach().numpy()
    
    return result
    