import tensorflow as tf
from Operator import *
from ImageOperators import *
from Graph import Graph
Data_Format='NHWC'

class QLearning():
	def __init__(self):
		pass
		
	def ConstructQFunc3D(self,ImageSize=5,BitDepth=7):
		self.IncidenceMatrix=tf.Variable("NetworkStructure")
		
		Conv3D_Layer1_Kernel=tf.placeholder([BitDepth,2,2,1,3])
		#[filter_depth, filter_height, filter_width, in_channels, out_channels]
		Conv3D_Layer1=tf.nn.conv3d(	input=self.IndcidenceMatrix,
									filter=Conv3D_Layer1_Kernel,
									strides=[1,1,1,1,1])
		
		Pooling2D_Layer1=tf.nn.max_pool(value=Conv3D_Layer1,
										ksize=[1,2,2,1],
										strides=[1,1,1,1])
		
		Conv2D_Layer2_Kernel=tf.placeholder([2,2,1,1])
		Conv2D_Layer2=tf.nn.conv2d(input=Pooling2D_Layer1,filter=Conv2D_Layer2_Kernel,strides=[1,1,1,1])
		
		Pooling2D_Layer2=tf.nn.max_pool(value=Conv3D_Layer2,
										ksize=[1,2,2,1],
										strides=[1,1,1,1])

		Conv2D_Layer3_Kernel=tf.placeholder([2,2,1,1])
		Conv2D_Layer3=tf.nn.conv2d(input=Pooling2D_Layer2,filter=Conv2D_Layer3_Kernel)
		
		Pooling2D_Layer3=tf.nn.max_pool(value=Conv3D_Layer3,
										ksize=[1,2,2,1],
										strides=[1,1,1,1])										
		Layer3_Shape=Pooling2D_Layer3.shape()
										
		Reshape_Layer3=tf.reshape(Pooling2D_Layer3,shape=[Layer3_Shape[0],-1])
		
		Dense_Layer4=tf.layers.dense(inputs=Reshape_Layer3, units=256, activation=tf.nn.relu)
		Dropout_Layer4=tf.layers.dropout(inputs=Dense_Layer4,rate=0.5)
		
		Output_Layer=tf.layers.dense(inputs=Reshape_Layer3, units=256, activation=tf.nn.sigmoid)
		
		self.Output=Output_Layer
	
	def SetNetworkGenerator(self,NetworkGenerator):
		self.NetworkGenerator=NetworkGenerator
	
	def SetOperatorList(self,OperatorList):
		self.OperatorList=OperatorList
	
	def StartTrial(self,TaskSpecific):
		LogHistory=TaskSpecific["LogHistory"]
		OperatorList=TaskSpecific["OperatorList"]
		NetworkGenerator=TaskSpecific["NetworkGenerator"]
		VertexNum=TaskSpecific["OperatorNum"]
		InputNum=TaskSpecific["InputNum"]
		OutputNum=TaskSpecific["OutputNum"]

		TaskInput=TaskSpecific["TaskInput"]
		for i in range(Times):
			Gph=Graph(	VertexNum=VertexNum,
						OperatorList=OperatorList,
						InputNum=InputNum,
						OutputNum=OutputNum
						)
			OptionList=Gph.ConnectionOptions()
			QNetInputList=[]
			for Option in OptionList:	
				Gph.ApplyOption(Option)
				QNetInput=Gph.UnifiedTransform('3D')
				QNetInputList.append(QNetInput)
				Gph.RevokeOption(Option)
			
			QValues=tf.train(self.Output,feed_dict={self.IncidenceMatrix:QNetInputList})
			_Sum=0
			for QValue in QValues:
				_Sum+=np.exp(QValue)
			ExpDist=[Qvalue/_Sum for QValue in QValues]
			ChosenOption=self.MakeChoice(Distribution=ExpDist,Choice=OptionList)
			
			Gph.ApplyOption(ChosenOption)
			BuiltNet=Gph.BuildGraph()
			Step=100
			Performance=self.TrainNet(BuiltNet,TaskInput,Step)
			
			HisItem={"OptionList":Gph.GetOptionList(),"TrainStep":Step,"Performance":Performance,"UnifiedNet":Gph.UnifiedTransform('3D')}
			if LogHistory==True:
				self.Log(HisItem)
			self.HisNet.append(HisItem["UnifiedNet"])
			self.HisNetPerf.append(HisItem["Performance"])
			QStep=100
			self.TrainQNet(self.Output,self.HisNet,self.HisNetPerf,QStep)
				
				
			
		

