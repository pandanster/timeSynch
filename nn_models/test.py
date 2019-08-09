from utility import createTrainTest,multiViewDataset,createLogger,computeAccuracy
from torch.utils.data import DataLoader
from model import lowResNet
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch

filePath='/mnt/d/impactProject/nn_models/train_test_all'
dirPath='/mnt/d/impactProject/ali_23words_out'
modelPath='/scratch/psanthal/multiView-All-Noclus/enc-model-360.pth'
classes=['teach','you','me','piano','want','very','angry','fire','everyone','huddle','how','today',
	'weather','wakeup','grandmother','never','there','actually','have','must','worried','they','visiting','students']
trainDataset=multiViewDataset(dirPath,classes,filePath)
saveDir='/scratch/psanthal/'
net=lowResNet(2048,len(classes),2,10,0,False,40)
net.load_state_dict(torch.load(model,map_location='cpu'),strict=False)
multiViewDataLoader=DataLoader(trainDataset,10,shuffle=True)
logger=createLogger('./','temp-log')
m=nn.Softmax(dim=1)
predictions=[]
labels=[]
for xy,yz,xz,label in multiViewDataLoader:		
	o=net({'xy':xy,'yz':yz,'xz':xz})
	predictions.append(torch.max(m(o),dim=1)[1].cpu().numpy().tolist()[0])
accuracy,confusion=computeAccuracy(labels,predictions,classes)
logger=createLogger('/scratch/psanthal','multiView-evaluation')
logger.info("The accuracy for separete thee model: %s is: %s",400,accuracy)
