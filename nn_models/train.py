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
dirPath='/mnt/d/time-synch/dataset-new'
trainTestFile='/mnt/d/time-synch/nn_models/train-test-split'
segmentfilepath='/mnt/d/time-synch/dataset_new.txt'
#createTrainTest(dirpath,segmentfilepath,outfilepath,classes)
#exit(0)
trainDataset=timeSynchDataset(dirPath,classes,trainTestFile,segment=True,segmentFile=segmentfilepath,Train=True,\
	skipsamplepercent=None,sensorDetail=sensorDetail	)
saveDir='/scratch/psanthal/'
'''
lowResnet Parameters
hidden_dim,class_size,num_layers,batch_size,dropout,use_cuda,frameCount
'''
net=IMUNet(1024,len(classes),1,10,0,False)
optimizer=optim.Adam(net.parameters(),lr=0.000001)
timeSynchDataLoader=DataLoader(trainDataset,10,shuffle=True,collate_fn=lambda x: x)
criterion=nn.CrossEntropyLoss()
logger=createLogger('./','temp-log')
for epoch in range(100):
	running_loss=0
	batchCount=0
	for data in timeSynchDataLoader:
		net.zero_grad()		
		y=net({'accl':[data[i][0] for i in range(len(data))],'laccl':[data[i][1] for i in range(len(data))],\
			'gyro':[data[i][2] for i in range(len(data))]})
		loss=criterion(y,torch.tensor([data[i][3] for i in range(len(data))],dtype=torch.long))
		loss.backward()
		optimizer.step()
		running_loss+=loss.item()
		batchCount+=1
		if batchCount==1:
			logger.info("Loss for epoch:%s is: %s",epoch,(running_loss/batchCount))
			batchCount=0
			running_loss=0
	if epoch%10==0 and epoch > 0:
		torch.save(net.state_dict(),saveDir+'model-'+str(epoch)+'.pth')

