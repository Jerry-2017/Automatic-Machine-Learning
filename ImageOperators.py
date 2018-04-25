import tensorflow as tf
from Graph import Operator

Data_Format='NHWC'

def ConcatImageTensor(TensorList):
	LenList=len(TensorList)
	BatchSize=TensorList[0].shape[0]
	PadTensor=[]
	MaxHeight=0
	MaxWidth=0
	for i in range(LenList):
		TensorShape=TensorList[i].shape()
		assert TensorShape[0]==BatchSize,'Batchsize inequal'
		if TensorShape[1]>MaxHeight:
			MaxHeight=TensorShape[1]
		if TensorShape[2]>MaxWidth:
			MaxWidth=TensorShape[2]
	for i in range(LenList):
		TensorShape=TensorList[i].shape()
		Paddings=[[0,0],[0,TensorShape[1]-MaxHeight],[0,TensorShape[2]-MaxWidth],[0,0]]
		PadTensor.append(tf.pad(TensorList[i]),Paddings,'SYMMETRIC')
	ConcatTensor=tf.concat(PadTensor,axis=3)
	return ConcatTensor

def ConcatOperator(OperatorList):
	TensorList=[]
	for Input in OperatorList:
		Tensor=Input.GetTensor()
		H,W,C=Input.GetImageAttr()
		if Tensor.shape()==(BatchSize,H,W,C):
			ReshapeTensor=Tensor
		else:
			ReshapeTensor=Tensor.reshape([BatchSize,H,W,C])
		TensorList.append(ReshapeTensor)
	return Tensor=ConcatImageTensor(TensorList)


class ImageOperator(Operator):
	def SetImageAttr(self,Width,Height,Channel):
		self.Width=Width
		self.Height=Height
		self.Channel=Channel
	def GetImageAttr(self):
		return self.Width,self.Height,self.Channel

class ImageInput(ImageOperator):
	def __init__(self,_Input):
		Super(ImageInput,self).__init__(InputList=[_Input],InputDegree=0,OutputDegree=1,Name='Input')
		
	def ConstructFunc(self,InputList):
		self.Tensor=InputList[0]		
		

def Conv2DFactory(Size,ChannelCoef,Stride):
	class Conv2D(ImageOperator):
		_Height=Size
		_Width=Size
		_Stride=Stride
		_ChannelCoef=ChannelCoef
		def __init__(self,Input):
			super(Conv2D,self).__init__(InputList=Input,
										InputDegree=1,
										OutputDegree=1,
										Name='Conv2D')
		def ConstructFunc(self,InputList):
			InputOp=InputList[0]
			InputTensor=InputOp.GetTensor()
			Shape=InputTensor.shape
			OutputChannelNum=int(Shape[3]*Conv2D._ChannelCoef)
			assert(OutputChannelNum>=1)
			Filter=tf.placeholder([Conv2D._Height,Conv2D._Width,Shape[3],OutputChannelNum)
			self.Tensor=tf.nn.conv2d(	input=InputTensor,
										filter=Filter,
										strides=Conv2D.Stride,
										padding='SAME',
										data_format=Data_Format)
			
			self.SetImageAttr(
		
	return Conv2D
	
def PoolingFactory(Size,Stride,Type):
	assert _Type in ['Max','Avg']
	class Pooling(ImageOperator):
		_Size=Size
		_Stride=Stride
		_Type=Type
		def __init__(self,Input):
			super(Pooling,self).__init__(	InputList=Input,
											InputDegree=1,
											OutputDegree=1,
											Name='Pooling')
		def ConstructFunc(self,InputList):
			InputOp=InputList[0]
			InputTensor=InputOp.GetTensor()
			if Pooling.Type=='Max':
				self.Tensor=tf.nn.max_pool(	InputTensor,
												Size=Pooling._Size,
												strides=Pooling._Stride,
												padding='SAME',
												data_format=Data_Format)
			elif Pooling.Type='Avg':
				self.Tensor=tf.nn.avg_pool(	InputTensor,
												Size=Pooling._Size,
												strides=Pooling._Stride,
												padding='SAME',
												data_format=Data_Format)
)
		
		
def TransConv2DFactory(Size,ImageCoef,ChannelCoef,Strides):
	class TransConv2D(ImageOperator):
		_ImageCoef=ImageCoef
		_Strides=Strides
		_Height=Size
		_Width=Size
		_ChannelCoef=ChannelCoef
		def __init__(self,Input):
			super(TransConv2D,self).__init__(InputList=Input,InputDegree=1,OutputDegree=1,Name='TransConv2D')
		def ConstructFunc(self,InputList):
			InputOp=InputList[0]
			InputTensor=InputOp.GetTensor()
			Shape=InputTensor.shape
			OutputChannelNum=int(Shape[3]*TransConv2D._ChannelCoef)
			OutputHeight=int(Shape[1]*TransConv2D._ImageCoef)
			OutputWidth=int(Shape[2]*TransConv2D._ImageCoef)
			Filter=tf.placeholder([TransConv2DFactory._Height,TransConv2DFactory._Width,Shape[3],OutputChannelNum]) """check"""
			OutputShape=[Shape[0],OutputHeight,OutputWidth.ImageCoef,OutputChannelNum]
			self.Tensor=tf.nn.conv2d_transpose(	value=InputTensor,
												filter=Filter,
												output_shape=OutputShape,
												strides=TransConv2D._Stride,
												padding='SAME',
												data_format=Data_Format)
	return TransConv2D	
	
def ActivationFactory(Type):
	class Activation(ImageOperator):
		_Type=Type
		def __init__(self,Input):
			super(Activation,self).__init__(InputList=[Input],InputDegree=1,OutputDegree=1,Name='Activation')
		def ConstructFunc(self,InputList):
			InputOp=InputList[0]
			InputTensor=InputOp.GetTensor()
			if Activation._Type=='Relu':
				self.Tensor=tf.nn.relu(features=InputTensor)
			elif  Activation._Type=='Leaky_Relu':
				self.Tensor=tf.nn.leaky_relu(features=InputTensor):
			elif  Activation._Type=='L2_Norm':
				self.Tensor=tf.nn.l2_normalize(features=InputTensor):
	return Activation
)

def BinaryOpFactory(Type):
	class BinaryOp(ImageOperator):
		_Type=Type
		def __init__(self,Input):
			super(BinaryOp,self).__init__(InputList=Input,InputDegree=2,OutputDegree=1,Name='BinaryOp')
		def ConstructFunc(self,InputList):
			InputOp1=InputList[0]
			InputOp2=InputList[1]
			InputOp1Tensor=InputOp1.GetTensor()
			InputOp2Tensor=InputOp2.GetTensor()
			assert InputOp1Tensor.shape==InputOp2Tensor.shape
			if BinaryOp._Type=='Concat':
				self.Tensor=ConcatImageTensor([InputOp1Tensor,InputOp2Tensor])
			elif BinaryOp._Type=='Add':
				self.Tensor=tf.add(InputOp1Tensor,InputOp2Tensor)
	return BinaryOp

def ReuseFactory(OutputputNum):
	class Reuse(ImageOperator):
		_OutputNum=OutputNum
		def __init__(self,Input):
			super(Reuse,self).__init__(InputList=Input,InputeDegree=1,OutputDegree=_OutputNum,Name='Reuse')
		def ConstructFunc(self,InputList):
			self.Tensor=InputList[0]
	return Reuse
			
	
def ConcatFactory(InputNum):
	class Concat(ImageOperator):
		_InputNum=InputNum
		def __init__(self,Input):
			super(Concat,self).__init__(InputList=Input,InputDegree=_InputNum,OutputDegree=1,Name='Concat')
		def ConstructFunc(self,InputList):
			self.Tensor=ConcatOperator(InputList)
	return Concat
	
def DenseFactory(HiddenNumCoef):
	class Dense(ImageOperator):
		_HiddenNumCoef=HiddenNumCoef
		def __init__(self,Input):
			super(Dense,self).__init__(InputList=Input,InputDegree=1,OutputDegree=1,Name='Dense')
		def ConstructFunc(self,InputList):
			Input=InputList[0]
			InputTensor=Input.GetTensor()
			InputTensor=tf.reshape(InputTensor,[BatchSize,-1])
            Height,Width,Channel=Input.GetImageAttr()
            Height=int(HiddenNumCoef*Height)
            Width=int(HiddenNumCoef*Width)
            
			self.SetImageAttr(Height,Width,Channel)
			self.Tensor=tf.dense(input=InputTensor,units=Height*Width*Channel)