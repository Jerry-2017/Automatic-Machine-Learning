import tensorflow as tf
from Graph import Operator

Data_Format='NHWC'

def Conv2DFactory(Size,ChannelCoef,Stride):
	class Conv2D(Operator):
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
	return Conv2D
	
def PoolingFactory(Size,Stride,Type):
	assert _Type in ['Max','Avg']
	class Pooling(Operator):
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
	class TransConv2D(Operator):
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
	class Activation(Operator):
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
	class BinaryOp(Operator):
		_Type=Type
		def __init__(self,Input):
			super(Activation,self).__init__(InputList=Input,InputDegree=2,OutputDegree=1,Name='Activation')
		def ConstructFunc(self,InputList):
			InputOp1=InputList[0]
			InputOp2=InputList[1]
			InputOp1Tensor=InputOp1.GetTensor()
			InputOp2Tensor=InputOp2.GetTensor()
			assert InputOp1Tensor.shape==InputOp2Tensor.shape
			if BinaryOp._Type=='Concat':
				self.Tensor=tf.concat([InputOp1Tensor,InputOp2Tensor],1
			elif BinaryOp._Type=='Add':
				self.Tensor=tf.add(InputOp1Tensor,InputOp2Tensor)
	return BinaryOp
				