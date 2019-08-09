import multiprocessing as mp
import glob
from torch.utils.data import Dataset
from string import digits
import numpy as np
import torch
import logging 
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
import random

def doMultiProcessing(inFunc,inDir,split,arguments,noJobs=16):
	processes=[]
	count=0
	inFiles=glob.glob(inDir+'/*')
	for i in range(0,len(inFiles),split):
		p = mp.Process(target=inFunc,args=tuple([inFiles[0:0+split]]+arguments))
		if count > noJobs:
			for k in processes:
				k.join()
				processes = []
				count = 0
		p.start()
		count += 1
	if count > 0:
		for k in processes:
			k.join()
	return

def rmDigit(label):
	rm_dig=str.maketrans('','',digits)
	return label.translate(rm_dig)

def getLower(inString):
	return inString.lower()

def getLabel(inFile,classes=None,withCount=True):
	if classes is not None:
		return classes[[cls in inFile for cls in classes].index(True)]
	user=inFile.strip().split('/')[-1].split('_')[0]
	name=inFile.strip().split('/')[-1]
	dirName=name.split('_')[1]+name.split('_')[2].split('.')[0]
	if withCount:
		return dirName
	else:
		return rmDigit(dirName)

def getUser(inFile):
	user=inFile.strip().split('/')[-1].split('_')[0]
	name=inFile.strip().split('/')[-1]
	return user

def getSegments(data,segments=None):
	if segments is not None:
		try:
			index=[segment[0] in data for segment in segments].index(True)
			return int(segments[index][1]),int(segments[index][2])
		except:
			return None,None

def getdata(infilepath,segmentFile=None):
	labels=[]
	files=[]
	indirs=glob.glob(infilepath+'/*')
	segments=open(segmentFile,'r')
	segments=segments.readlines()
	segments=[segment.strip().split(',') for segment in segments]
	for indir in indirs:
		label=indir.strip().split('/')[-1]
		count=0
		infiles=glob.glob(indir+'/*')
		for infile in infiles:
			f=pd.read_csv(infile,skiprows=1,header=None)
			start,end=getSegments(infile,segments)
			if start is None:
				print(infile)
				continue
			labels.append(label)
			files.append(infile.split('/')[-1])		
	return labels,files	

def createTrainTest(infilepath,segmentfile,outfile,classes,testsamples=10):
	labels,files=getdata(infilepath,segmentfile)
	cls_count={}
	test=[]
	train=[]
	outfile=open(outfile,'w')
	for cls in classes:
		cls_count[cls]=0
	for i,label in enumerate(labels):
		if label not in classes:
			continue
		if cls_count[label]<testsamples:
			test.append(i)
			cls_count[label]+=1
		else:
			train.append(i)
	train_files=np.array(files)[train]
	test_files=np.array(files)[test]
	write_file=lambda outfile,file_to_write,test_or_train: outfile.write(file_to_write+','+test_or_train+'\n')
	for train_file in train_files:
		write_file(outfile,train_file,'Train')
	for test_file in test_files:
		write_file(outfile,test_file,'Test')

def getSkippedData(labels,sensorDetail,classes,currentIndex,skipAmount):
	sensorToSearch=None
	for value in sensorDetail.values():
		if classes[labels[currentIndex]] in value:
			sensorToSearch=[classes.index(v) for v in value]
			break
	if sensorToSearch is None:
		print(classes[labels[currentIndex]])
		exit(0)
	labelToLook=sensorToSearch[random.randint(0,len(sensorToSearch)-1)]
	for i,label in enumerate(labels):
		if i != currentIndex and label==labelToLook:
			return i
	return None


def getSkipped(accl_data,laccl_data,gyro_data,labels,sensorDetail,classes,skipsamplepercent,time,fromData=False):
	skip_accl=[]
	skip_laccl=[]
	skip_gyro=[]
	skipTime=[]
	for i,data in enumerate(zip(accl_data,laccl_data,gyro_data)):
		accl=data[0]
		laccl=data[1]
		gyro=data[2]
		skipped=int((skipsamplepercent/100)*accl.shape[0])
		toSkip=int(accl.shape[0]-skipped)
		skipTime.append(time[i][accl.shape[0]-1]-time[i][toSkip])
		if fromData:
			skipIndex=getSkippedData(labels,sensorDetail,classes,i,skipped)
			skip_accl.append(np.concatenate((accl_data[skipIndex][0:skipped],accl[0:toSkip]),axis=0))
			skip_laccl.append(np.concatenate((laccl_data[skipIndex][0:skipped],laccl[0:toSkip]),axis=0))
			skip_gyro.append(np.concatenate((gyro_data[skipIndex][0:skipped],gyro[0:toSkip]),axis=0))

		else:
			skip_accl.append(np.concatenate((np.zeros((skipped,3))+.000000000000000001,accl[0:toSkip]),axis=0))
			skip_laccl.append(np.concatenate((np.zeros((skipped,3))+.000000000000000001,laccl[0:toSkip]),axis=0))
			skip_gyro.append(np.concatenate((np.zeros((skipped,3))+.000000000000000001,gyro[0:toSkip]),axis=0))

	return skipTime,skip_accl,skip_laccl,skip_gyro

class timeSynchDataset(Dataset):
	def __init__(self,dirPath,classes,trainTestFile=None,frameCount=40,logger=None,segment=False,skipsamplepercent=None,segmentFile=None,Train=True,sensorDetail=None):
		self.dirPath=dirPath
		self.classes=classes
		self.fileList=[]
		self.data=None
		self.skipData=None
		self.labels=[]
		self.frameCount=frameCount
		self.logger=logger
		indirs=glob.glob(dirPath+'/*')
		f=open(trainTestFile)
		f=f.readlines()
		f=[f.strip().split(',') for f in f]
		if Train:
			f=[f[0] for f in f if f[1] == 'Train']
		else:
			f=[f[0] for f in f if f[1] == 'Test']
		self.fileList=f
		self.data,self.labels=self.loaddata(indirs,classes,skipsamplepercent,segment,segmentFile,frameCount)
		if skipsamplepercent is not None:
			time_data,accl_data,laccl_data,gyro_data=getSkipped(self.data['accl'],self.data['laccl'],self.data['gyro'],\
				self.labels,sensorDetail,self.classes,skipsamplepercent,self.data['time'],True)
			self.skipData ={"accl":accl_data,"laccl":laccl_data,"gyro":gyro_data,"time":time_data}
	def __len__(self):
		return len(self.labels)
		
	def __getitem__(self,idx):
		if self.skipData is not None:
			return torch.tensor(self.skipData['accl'][idx],dtype=torch.float32),torch.tensor(self.skipData['laccl'][idx],\
				dtype=torch.float32),torch.tensor(self.skipData['gyro'][idx],dtype=torch.float32),torch.tensor(self.labels[idx],dtype=torch.long)
		else:
			return torch.tensor(self.data['accl'][idx],dtype=torch.float32),torch.tensor(self.data['laccl'][idx],dtype=torch.float32),\
				torch.tensor(self.data['gyro'][idx],dtype=torch.float32),torch.tensor(self.labels[idx],dtype=torch.long)
	
	def loaddata(self,indirs,classes,skipsamplepercent=None,segment=False,segmentFile=None,frameCount=None):
		labels=[]
		accel_data=[]
		linear_accel_data=[]
		gyro_data=[]
		time_data=[]
		if segment:
			segments=open(segmentFile,'r')
			segments=segments.readlines()
			segments=[segment.strip().split(',') for segment in segments]
		for indir in indirs:
			label=indir.strip().split('/')[-1]
			count=0
			infiles=glob.glob(indir+'/*')
			for infile in infiles:
				if infile.split('/')[-1] not in self.fileList:
					continue
				f=pd.read_csv(infile,skiprows=1,header=None)
				if segment:
					start,end=getSegments(infile,segments)
					if start is None:
						print(infile)
						continue
					f=f.iloc[start:end,:]
				time_data.append([k[0] for k in f.iloc[:,9:10].to_numpy().tolist()])
				f=f.iloc[:,0:9]
				f=f.to_numpy()
				labels.append(classes.index(label))
				accel_data.append(f[:,0:3])
				linear_accel_data.append(f[:,3:6])
				gyro_data.append(f[:,6:9])

		return {"accl":accel_data,"laccl":linear_accel_data,"gyro":gyro_data,"time":time_data},labels

def createLogger(inDir,logFile):
	logging.basicConfig(level=logging.INFO,
		format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
		handlers=[logging.FileHandler("{0}/{1}.log".format(inDir, logFile)),logging.StreamHandler()])
	return logging.getLogger()

def computeAccuracy(labels,predictions,classes):
	return confusion_matrix(labels,predictions,classes),accuracy_score(labels,predictions)