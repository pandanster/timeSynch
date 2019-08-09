import torch
import torch.nn as nn
import torch.nn.functional as F


def getConv(dimension,inChannel,outChannel,convW,convH,padding,useCuda):
	if useCuda:
		if dimension==1:
			return nn.Conv1d(inChannel,outChannel,(convW,convH),padding=padding).cuda()
		elif dimension==2:
			return nn.Conv2d(inChannel,outChannel,(convW,convH),padding=padding).cuda()
		elif dimension==3:
			return nn.Conv3d(inChannel,outChannel,(convW,convH),padding=padding).cuda()
	else:
		if dimension==1:
			return nn.Conv1d(inChannel,outChannel,(convW,convH),padding=padding)
		elif dimension==2:
			return nn.Conv2d(inChannel,outChannel,(convW,convH),padding=padding)
		elif dimension==3:
			return nn.Conv3d(inChannel,outChannel,(convW,convH),padding=padding)

def getLinear(inUnits,outUnits,useCuda):
	if useCuda:
		return nn.Linear(inUnits,outUnits).cuda()
	else:	
		return nn.Linear(inUnits,outUnits)

def getLSTM(inUnits,hiddenUnits,useCuda):
	if useCuda:
		return nn.LSTM(inUnits,hiddenUnits).cuda()
	else:
		return nn.LSTM(inUnits,hiddenUnits)

def getDropout(p,useCuda):
	if useCuda:
		return nn.Dropout(p=p).cuda()
	else:
		return nn.Dropout(p=p)
