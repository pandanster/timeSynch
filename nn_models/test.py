from utility import createTrainTest, timeSynchDataset,createLogger
from torch.utils.data import DataLoader
from model import IMUNet
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
classes=['Forward', 'Front-block', 'Kick', 'Lower-block', 'Lower-punch', \
'Raise','Side-block','Straight-punch', 'Upper-block','Upper-cut','Upper-punch']
sensorDetail={'right-hand':['Lower-punch','Straight-punch','Upper-punch','Upper-cut'], \
'left-hand':['Front-block','Side-block','Upper-block','Lower-block'],'right-leg':['Forward','Raise','Kick']}
#createTrainTest(userDirs,users,testCount,outFile,classes)
dirPath='/home/psanthal/timeSynch/dataset-new'
trainTestFile='/home/psanthal/timeSynch/nn_models/train-test-split'
segmentfilepath='/home/psanthal/timeSynch/dataset_new.txt'
#createTrainTest(dirpath,segmentfilepath,outfilepath,classes)
#exit(0)
testDataset=timeSynchDataset(dirPath,classes,trainTestFile,segment=True,segmentFile=segmentfilepath,Train=False,\
	skipsamplepercent=None,sensorDetail=sensorDetail	)

'''
lowResnet Parameters
hidden_dim,class_size,num_layers,batch_size,dropout,use_cuda,frameCount
'''
modelPath='/scratch/psanthal/timeSynch/model-70.pth'
net=IMUNet(1024,len(classes),1,10,0,False)
net.load_state_dict(torch.load(modelPath,map_location='cpu'),strict=False)
timeSynchDataLoader=DataLoader(trainDataset,10,shuffle=True,collate_fn=lambda x: x)
logger=createLogger('/scratch/psanthal/','timeSynch-Evaluation-log')
logger.info("Testing with skip percent: 0 for samples:%s",len(testDataset))
net.eval()
labels=[]
predictions=[]
m=nn.Softmax(dim=1)
for epoch in range(100):
	running_loss=0
	batchCount=0
	for data in timeSynchDataLoader:
		net.zero_grad()		
		y=net({'accl':[data[i][0] for i in range(len(data))],'laccl':[data[i][1] for i in range(len(data))],\
			'gyro':[data[i][2] for i in range(len(data))]})
		predictions+=torch.max(m(y),dim=1)[1].cpu().numpy().tolist()
		labels=[data[i][3].cpu().numpy().tolist()[0] for i in range(len(data))]
confusion,accuracy=computeAccuracy(labels,predictions,[i for i in range(len(classes))])
logger.info("The accuracy for model: %s for 0 skip is: %s",70,accuracy)

