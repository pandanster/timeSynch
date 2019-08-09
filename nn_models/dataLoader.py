import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import glob
import torch.optim as optim
from string import digits
import logging
import logging.handlers

def localize(x,use_cuda):
	if use_cuda:
		return x.cuda()
	else:
		return x

def getTargetPadded(targets,classes):
	maxLen=max([len(target) for target in targets])
	targets=[[classes.index(cls) for cls in target] for target in targets]
	paddedTargets=np.full((len(targets),maxLen),fill_value=(len(classes)-1),dtype='i')
	for e,l in enumerate(targets):
		paddedTargets[e,:len(l)]=l
	return paddedTargets

def saveNumpy(inFiles,outDir,frameCount,logger):
	views=['xy','yz','xz']
	bodyParts=['left','right','body']
	for inFile in inFiles:
		label=inFile.strip().split('/')[-1]
		viewsData={'xy':[],'yz':[],'xz':[]}
		for view in views:
			for part in bodyParts:
				frames=glob.glob(inFile+'/'+view+'/'+part+'/*')
				print(len(frames))
				frames=np.array([np.array(Image.open(frame)) for frame in frames])
				if frames.shape[0] < frameCount:
					logger.info("For files %s the frame count is %s",inFile,frames.shape[0])
					continue
				elif frames.shape[0] > frameCount:
					start=int(2*(frames.shape[0]-frameCount)/3)
				else:
					start=0
				frames=frames[start+start+frameCount]
				viewsData[view].append[frames]
			np.save(outDir+'/'+label+'_'+view,np.array(viewsData[view]))

def getDataNumpy(inDir,classes,logger,saveDir=None,user=None,nonMan=None):
	inFiles=glob.glob(inDir+'/*')	
	labels=[]
	body=[]
	left=[]
	right=[]
	frameCount=40
	inFiles=sorted(inFiles)
	for i in range (0,len(inFiles),3):
		body_name=inFiles[i].split('/')[-1]
		left_name=inFiles[i].split('/')[-1]
		right_name=inFiles[i].split('/')[-1]
		bodyData=np.load(inFiles[i])
		if bodyData.shape[0] < frameCount:
			logger.info("For files %s the frame count is %s",inFiles[i],bodyData.shape[0])
			continue
		elif bodyData.shape[0] > frameCount:
			start=int(2*(bodyData.shape[0]-frameCount)/3)
		else:
			start=0
		try:
			labels.append([cls in body_name for cls in classes].index(True))
		except:
			print("Lable not found for {}".format(inFiles[i]))
			continue

		body.append(np.load(inFiles[i])[start:start+frameCount])
		left.append(np.load(inFiles[i+1])[start:start+frameCount])
		right.append(np.load(inFiles[i+2])[start:start+frameCount])
	if saveDir is None:
		return np.array(body),np.array(left),np.array(right),labels
	else:
		savePickle(inDir,saveDir,user,classes,loaded={'body':np.array(body),'left':np.array(left),'right':np.array(right),'label':np.array(labels),'files':np.array(inFiles)})	
			

def getData(inDir,classes,targets,logger,nonMan=None):
	files=glob.glob(inDir+'/*')
	data_xy_left=[]
	data_xy_body=[]
	data_xy_right=[]
	data_yz_body=[]
	data_yz_left=[]
	data_yz_right=[]
	data_xz_right=[]
	data_xz_left=[]
	data_xz_body=[]
	nonMan_labels=[]
	labels=[]
	targetLens=[]
	rm_dig=str.maketrans('','',digits)
	shape=None
	base=100
	chosenLen=60
	consideredParts=['left','right','body']
	for file in files:
		temp_xy={'left':[],'body':[],'right':[]}
		temp_yz={'left':[],'body':[],'right':[]}
		temp_xz={'left':[],'body':[],'right':[]}
		try:
			label=file.strip().split('/')[-1]
			label=label.translate(rm_dig)
			if "nonM" in label:
				if "Wake" in label:
					nonMan_labels.append(nonMan.index("negate"))
				elif "Worried" in label:
					nonMan_labels.append(nonMan.index("assert"))
				elif "IWant" in label:
					nonMan_labels.append(nonMan.index("backLean"))
				elif "One" in label:
					nonMan_labels.append(nonMan.index("sideLean"))
				elif "How" in label:
					nonMan_labels.append(nonMan.index("frontLean"))
			else:
				nonMan_labels.append(nonMan.index("Manual"))	
			if "DontWake" in label:
				label="mewakeup"
			elif "Worried" in label:
				label="meworried"
			elif "IWant" in label:
				label="mewantpiano"
			elif "One" in label:
				label="onepianotwobooks"
		except:
			logger.info("Failed to parse label for file:%s",file)
			continue
		label=label.lower()
		if any([all([cls in label for cls in target]) for target in targets]):
			label=targets[[all([cls in label for cls in target]) for target in targets].index(True)]
		else:	
			logger.info("Label:%s not found for file:%s",label,file)
			continue
		views=glob.glob(file+'/*')
		for view in views:
			currView=view.strip().split('/')[-1]
			bodyParts=glob.glob(view+'/*')
			for part in bodyParts:
				currPart=part.strip().split('/')[-1]
				imFiles=glob.glob(part+'/*')	
				imFiles=sorted(imFiles)
				for imFile in imFiles:
					im=Image.open(imFile)
					im=np.asarray(im)
					if 'xy' in currView:
						temp_xy[currPart].append(im)
					if 'yz' in currView:
						temp_yz[currPart].append(im)
					if 'xz' in currView:
						temp_xz[currPart].append(im)
					if shape is None:
						shape=im.shape
					if shape[0] != im.shape[0] or shape[1] != im.shape[1]:
						logger.info("Image shape is different %s",file)
		if len(temp_xy['left']) != len(temp_yz['left']) or len(temp_yz['left']) != len(temp_xz['left']) or len(temp_xy['left']) ==0 or len(temp_xy['left']) < chosenLen or any([len(temp_xy[part])!=len(temp_xy['left']) for part in consideredParts]):
	#	if len(temp_xy) != len(temp_yz) or len(temp_yz) != len(temp_xz) or len(temp_xy) ==0:
			logger.info("Different length in file:%s, xy:%s, yz:%s, xz:%s,body:%s,right:%s",file,len(temp_xy['left']),len(temp_yz['right']),len(temp_xz['left']),len(temp_xy['body']),len(temp_xy['right']))
			continue
		elif len(temp_xy['left']) > chosenLen:
			exceeding=len(temp_xy['left'])-chosenLen
			startLen=int(2*(exceeding/3))
			endLen=startLen+60
			for part in consideredParts:
				temp_xy[part]=temp_xy[part][startLen:endLen]
				temp_yz[part]=temp_yz[part][startLen:endLen]
				temp_xz[part]=temp_xz[part][startLen:endLen]
		data_xy_body.append(temp_xy['body'])
		data_xy_left.append(temp_xy['left'])
		data_xy_right.append(temp_xy['right'])
		data_yz_body.append(temp_yz['body'])
		data_yz_left.append(temp_yz['left'])
		data_yz_right.append(temp_yz['right'])
		data_xz_body.append(temp_xz['body'])
		data_xz_left.append(temp_xz['left'])
		data_xz_right.append(temp_xz['right'])
		labels.append(label)
		targetLens.append(len(label))
	return files,np.array(data_xy_body),np.array(data_xy_left),np.array(data_xy_right),np.array(data_yz_body),np.array(data_yz_left),np.array(data_yz_right),np.array(data_xz_body),np.array(data_xz_left),np.array(data_xz_right),np.array(getTargetPadded(labels,classes)),np.array(targetLens),np.array(nonMan_labels)

			
def savePickle(inDir,outDir,forUser,classes,targets=None,logger=None,nonMan=None,loaded=None):
	if loaded is None:
		files,xy_body,xy_left,xy_right,yz_body,yz_left,yz_right,xz_body,xz_left,xz_right,labels,target_len,nonMan_labels=getData(inDir,classes,targets,logger,nonMan)
		np.save(outDir+forUser+"_xy_body.npy",xy_body)
		np.save(outDir+forUser+"_xy_left.npy",xy_left)
		np.save(outDir+forUser+"_xy_right.npy",xy_right)
		np.save(outDir+forUser+"_yz_body.npy",yz_body)
		np.save(outDir+forUser+"_yz_left.npy",yz_left)
		np.save(outDir+forUser+"_yz_right.npy",yz_right)
		np.save(outDir+forUser+"_xz_body.npy",xz_body)
		np.save(outDir+forUser+"_xz_left.npy",xz_left)
		np.save(outDir+forUser+"_xz_right.npy",xz_right)
		np.save(outDir+forUser+"_labels.npy",labels)
		np.save(outDir+forUser+"_files.npy",np.array(files))
		np.save(outDir+forUser+"_target_len.npy",target_len)
		np.save(outDir+forUser+"_nonM_labels.npy",nonMan_labels)
	else:
		np.save(outDir+forUser+"_body.npy",loaded['body'])
		np.save(outDir+forUser+"_left.npy",loaded['left'])
		np.save(outDir+forUser+"_right.npy",loaded['right'])
		np.save(outDir+forUser+"_files.npy",loaded['files'])
		np.save(outDir+forUser+"_labels.npy",loaded['label'])
		
def loadPickle(inDir,forUser,multiView):
	if multiView:	
		xy_body=np.load(inDir+forUser+"_xy_body.npy")
		xy_left=np.load(inDir+forUser+"_xy_left.npy")
		xy_right=np.load(inDir+forUser+"_xy_right.npy")
		yz_body=np.load(inDir+forUser+"_yz_body.npy")
		yz_left=np.load(inDir+forUser+"_yz_left.npy")
		yz_right=np.load(inDir+forUser+"_yz_right.npy")
		xz_body=np.load(inDir+forUser+"_xz_body.npy")
		xz_left=np.load(inDir+forUser+"_xz_left.npy")
		xz_right=np.load(inDir+forUser+"_xz_right.npy")
		labels=np.load(inDir+forUser+"_labels.npy")
		targetLen=np.load(inDir+forUser+"_target_len.npy")
		nonMan_labels=np.load(inDir+forUser+"_nonM_labels.npy")
		files=np.load(inDir+forUser+"_files.npy")
		return xy_body,xy_left,xy_right,yz_body,yz_left,yz_right,xz_body,xz_left,xz_right,labels,files,targetLen,nonMan_labels
	else:
		body=np.load(inDir+forUser+"_body.npy")
		left=np.load(inDir+forUser+"_left.npy")
		right=np.load(inDir+forUser+"_right.npy")
		labels=np.load(inDir+forUser+"_labels.npy")
		files=np.load(inDir+forUser+"_files.npy")
		return body,left,right,labels,files
		
	
def getMultiDirData(inDir,userDirs,classes,logger,pickled=False):
	data_xy=None
	data_yz=None
	data_xz=None
	labels=None
	files=None
	for userDir in userDirs:
		if pickled:
			temp_xy,temp_yz,temp_xz,temp_labels,temp_files=loadPickle(inDir,userDir)
		else:	
			temp_files,temp_xy,temp_yz,temp_xz,temp_labels=getData(inDir+'/'+userDir,classes,logger)
		if data_xy is None:
			data_xy=temp_xy
			data_yz=temp_yz
			data_xz=temp_xz
			labels=temp_labels
			files=temp_files
		else:
			data_xy=np.concantenate((data_xy,temp_xy),axis=0)
			data_yz=np.concantenate((data_yz,temp_yz),axis=0)
			data_xz=np.concantenate((data_xz,temp_xz),axis=0)
			labels=temp_labels
			files=files+temp_files
	return data_xy,data_yz,data_xa,labels,files

def getTrainTest(inDir,users,classes,testCount,indClasses=None,nonManOnly=False,multiView=False):
	countKepper={}
	userCount={}
	if multiView:
		train_xy_body=None
		train_xy_left=None
		train_xy_right=None
		train_yz_body=None
		train_yz_left=None
		train_yz_right=None
		train_xz_body=None
		train_xz_left=None
		train_xz_right=None
		train_files=None
		trainClass=[]
		trainNmClass=[]
		test_xy_body=None
		test_xy_left=None
		test_xy_right=None
		test_yz_body=None
		test_yz_left=None
		test_yz_right=None
		test_xz_body=None
		test_xz_left=None
		test_xz_right=None
		test_files=None
		testClass=[]
		testNmClass=[]
		targets=classes
		classes=['_'.join(cls) for cls in classes]
	else:
		train_body=None
		train_xy_left=None
		train_xy_right=None
		trainClass=[]
		test_body=None
		test_xy_left=None
		test_xy_right=None
		testClass=[]
	for cls in classes:
		countKepper[(cls)]=0
	for user in users:
		for cls in classes:
			userCount[cls]=0
		train_labels=[]
		train_indices=[]
		test_indices=[]
		test_labels=[]
		if multiView:
			xy_body,xy_left,xy_right,yz_body,yz_left,yz_right,xz_body,xz_left,xz_right,labels,files,targetLen,nonMan_labels=loadPickle(inDir,user,multiView)
		else:
			body,left,right,labels,files=loadPickle(inDir,user,multiView)
		if nonManOnly:
			labels=nonMan_labels
		for i,label in enumerate(labels.tolist()):
			if label > len(classes)-1:
				continue
			if type(label) is list:
				label='_'.join([indClasses[l] for l in label][:targetLen[i]])
				label=classes.index(label)
			if userCount[classes[label]] < 2 and countKepper[classes[label]] < testCount:
				test_labels.append(labels[i])
				test_indices.append(i)
				userCount[classes[label]]+=1
				countKepper[classes[label]]+=1	
			else:
				train_indices.append(i)
				train_labels.append(labels[i])
		if multiView:
			if train_xy_body is None:
				train_xy_body=xy_body[train_indices]
				train_xy_left=xy_left[train_indices]
				train_xy_right=xy_right[train_indices]
				train_yz_body=yz_body[train_indices]
				train_yz_left=yz_left[train_indices]
				train_yz_right=yz_right[train_indices]
				train_xz_body=xz_body[train_indices]
				train_xz_left=xz_left[train_indices]
				train_xz_right=xz_right[train_indices]
				train_files=files[train_indices]
				train_tgt_len=targetLen[train_indices]
				train_nm_class=nonMan_labels[train_indices]	
			else:
				train_xy_body=np.concatenate((train_xy_body,xy_body[train_indices]),axis=0)
				train_xy_left=np.concatenate((train_xy_left,xy_left[train_indices]),axis=0)
				train_xy_right=np.concatenate((train_xy_right,xy_right[train_indices]),axis=0)
				train_yz_body=np.concatenate((train_yz_body,yz_body[train_indices]),axis=0)
				train_yz_left=np.concatenate((train_yz_left,yz_left[train_indices]),axis=0)
				train_yz_right=np.concatenate((train_yz_right,yz_right[train_indices]),axis=0)
				train_xz_body=np.concatenate((train_xz_body,xz_body[train_indices]),axis=0)
				train_xz_left=np.concatenate((train_xz_left,xz_left[train_indices]),axis=0)
				train_xz_right=np.concatenate((train_xz_right,xz_right[train_indices]),axis=0)
				train_files=np.concatenate((train_files,files[train_indices]),axis=0)
				train_tgt_len=np.concatenate((train_tgt_len,targetLen[train_indices]),axis=0)
				train_nm_class=np.concatenate((train_nm_class,train_nm_class[train_indices]),axis=0)
			if test_xy_body is None:
				test_xy_body=xy_body[test_indices]
				test_xy_left=xy_left[test_indices]
				test_xy_right=xy_right[test_indices]
				test_yz_body=yz_body[test_indices]
				test_yz_left=yz_left[test_indices]
				test_yz_right=yz_right[test_indices]
				test_xz_body=xz_body[test_indices]
				test_xz_left=xz_left[test_indices]
				test_xz_right=xz_right[test_indices]
				test_files=files[test_indices]
				test_tgt_len=targetLen[test_indices]
				test_nm_class=nonMan_labels[test_indices]	
			else:
				test_xy_body=np.concatenate((test_xy_body,xy_body[test_indices]),axis=0)
				test_xy_left=np.concatenate((test_xy_left,xy_left[test_indices]),axis=0)
				test_xy_right=np.concatenate((test_xy_right,xy_right[test_indices]),axis=0)
				test_yz_body=np.concatenate((test_yz_body,yz_body[test_indices]),axis=0)
				test_yz_left=np.concatenate((test_yz_left,yz_left[test_indices]),axis=0)
				test_yz_right=np.concatenate((test_yz_right,yz_right[test_indices]),axis=0)
				test_xz_body=np.concatenate((test_xz_body,xz_body[test_indices]),axis=0)
				test_xz_left=np.concatenate((test_xz_left,xz_left[test_indices]),axis=0)
				test_xz_right=np.concatenate((test_xz_right,xz_right[test_indices]),axis=0)
				test_files=np.concatenate((test_files,files[test_indices]),axis=0)
				test_tgt_len=np.concatenate((test_tgt_len,files[test_tgt_len]),axis=0)
				test_nm_class=np.concatenate((test_nm_class,test_nm_class[test_indices]),axis=0)
		else:
			if train_body is None:
				train_body=body[train_indices]
				train_left=left[train_indices]
				train_right=right[train_indices]
				train_files=files[train_indices]
			else:
				train_body=np.concatenate((train_body,body[train_indices]),axis=0)
				train_left=np.concatenate((train_left,left[train_indices]),axis=0)
				train_right=np.concatenate((train_right,right[train_indices]),axis=0)
				train_files=np.concatenate((train_files,files[train_indices]),axis=0)
			if test_body is None:
				test_body=body[test_indices]
				test_left=left[test_indices]
				test_right=right[test_indices]
				test_files=files[test_indices]
			else:
				test_body=np.concatenate((test_body,body[test_indices]),axis=0)
				test_left=np.concatenate((test_left,left[test_indices]),axis=0)
				test_right=np.concatenate((test_right,right[test_indices]),axis=0)
				test_files=np.concatenate((test_files,files[test_indices]),axis=0)

				
					
		trainClass+=train_labels
		testClass+=test_labels
	
	if any([test_file in train_files.tolist() for test_file in test_files.tolist()]):
		logger.info("Test Files overlap with train files")
		exit(0)
	if multiView:	
		return train_xy_body,train_xy_left,train_xy_right,train_yz_body,train_yz_left,train_yz_right,train_xz_body,train_xz_left,train_xz_right,trainClass,train_tgt_len,train_nm_class,test_xy_body,test_xy_left,test_xy_right,test_yz_body,test_yz_left,test_yz_right,test_xz_body,test_xz_left,test_xz_right,testClass,test_tgt_len,test_nm_class
	else:
		return train_body,train_left,train_right,trainClass,train_files,test_body,test_left,test_right,testClass,test_files
			
