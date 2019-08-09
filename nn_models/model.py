from networks import getConv,getLSTM,getLinear,getDropout
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence

class IMUNet(nn.Module):
	def __init__(self,hidden_dim,class_size,num_layers,batch_size,dropout,use_cuda):
		super(IMUNet,self).__init__()
		self.hidden_dim=hidden_dim
		self.use_cuda=use_cuda
		self.class_size=class_size
		self.num_layers=num_layers
		self.modules={}
		self.lstm={}
		self.devices=['accl','laccl','gyro']
		for device in self.devices:
			self.lstm[device]=self.localize(nn.LSTM(input_size=3,hidden_size=hidden_dim,num_layers=self.num_layers,batch_first=True))
		self.linear=self.localize(nn.Sequential(nn.Linear(self.hidden_dim*3,512),nn.Dropout(p=dropout),nn.Linear(512,class_size)))
		self.batch=batch_size
		self.num_layers=num_layers

	def init_hidden(self,batch):
		return (self.localize(torch.zeros(1,batch,self.hidden_dim)),self.localize(torch.zeros(1,batch,self.hidden_dim)))

	def forward(self,x):
		self.batch=len(x['accl'])
		finalLayer=None
		for device in self.devices:
			x1=self.localize(x[device])
			lens=[x1[i].shape[0] for i in range(self.batch)]
			x1=pad_sequence(x1,batch_first=True)
			x1=pack_padded_sequence(x1,lens,batch_first=True,enforce_sorted=False)		
			o,h=self.lstm[device](x1)
			o=pad_packed_sequence(o,batch_first=True)[0]
			o=torch.cat([o[i,length-1] for i,length in enumerate(lens)]).view(self.batch,-1)
			print(o.shape)
			if finalLayer is None:
				finalLayer=o
			else:
				finalLayer=torch.cat((finalLayer,o),dim=1)
		return self.linear(finalLayer)

	def localize(self,x):
		if self.use_cuda:
			return x.cuda()
		else:
			return x
