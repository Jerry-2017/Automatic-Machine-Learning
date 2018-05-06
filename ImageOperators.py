import tensorflow as tf
from Graph import Operator

Data_Format='NHWC'

def ConcatImageTensor(TensorList):
    LenList=len(TensorList)
    BatchSize=TensorList[0].get_shape().as_list()[0]
    PadTensor=[]
    MaxHeight=0
    MaxWidth=0
    for i in range(LenList):
        TensorShape=TensorList[i].get_shape().as_list()
        assert TensorShape[0] is None or TensorShape[0]==BatchSize,'Batchsize inequal'
        if TensorShape[1]>MaxHeight:
            MaxHeight=TensorShape[1]
        if TensorShape[2]>MaxWidth:
            MaxWidth=TensorShape[2]
    for i in range(LenList):
        TensorShape=TensorList[i].get_shape().as_list()
        Paddings=[[0,0],[0,TensorShape[1]-MaxHeight],[0,TensorShape[2]-MaxWidth],[0,0]]
        PadTensor.append(tf.pad(TensorList[i],Paddings,'SYMMETRIC'))
    ConcatTensor=tf.concat(PadTensor,axis=3)
    return ConcatTensor

def ConcatOperator(OperatorList):
    TensorList=[]
    for Input in OperatorList:
        print(Input)
        Tensor=Input.GetTensor()
        TensorShape=Tensor.get_shape().as_list()
        H,W,C=Input.GetImageAttr()
        BatchSize=TensorShape[0]
        if TensorShape==(TensorShape[0],H,W,C):
            ReshapeTensor=Tensor
        else:
            ReshapeTensor=tf.reshape(Tensor,[-1,H,W,C])
        TensorList.append(ReshapeTensor)
    Tensor=ConcatImageTensor(TensorList)
    return Tensor


class ImageOperator(Operator):
    def __init__(self,InputList,ID,BuildFlag=True):
        if len(InputList)==1 and InputList[0] is ImageOperator:
            self.SetImageAttr(InputList[0].GetImageAttr())
        super(ImageOperator,self).__init__(InputList=InputList,ID=ID,BuildFlag=BuildFlag)
    
    def SetImageAttr(self,Height,Width,Channel):
        self.Width=Width
        self.Height=Height
        self.Channel=Channel
    def GetImageAttr(self):
        return self.Width,self.Height,self.Channel

    def CheckValid(self,InputList):
        return True

class ImageInput(ImageOperator):
    InputDegree=0
    OutputDegree=1
    Name='Input'
    def __init__(self,InputList,ID,BuildFlag=True):
        super(ImageInput,self).__init__(InputList=InputList,ID=ID,BuildFlag=BuildFlag)
        
    def ConstructFunc(self,InputList):
        self.Tensor=InputList[0]      
        Shape=self.Tensor.get_shape().as_list()
        self.SetImageAttr(*Shape[1:])

def Conv2DFactory(Size,ChannelCoef,Stride):
    class Conv2D(ImageOperator):
        _Height=Size
        _Width=Size
        _Stride=Stride
        _ChannelCoef=ChannelCoef
        InputDegree=1
        OutputDegree=1
        Name='Conv2D'
        def __init__(self,InputList,ID,BuildFlag=True):
            super(Conv2D,self).__init__(InputList=InputList,ID=ID,BuildFlag=BuildFlag)
        def CheckInputList(self,InputList):
            InputOp=InputList[0]
            #_Width
        def ConstructFunc(self,InputList):
            InputOp=InputList[0]
            InputTensor=InputOp.GetTensor()
            #print(InputTensor)
            Shape=InputTensor.get_shape().as_list()
            #print(Shape)
            OutputChannelNum=int(Shape[3]*Conv2D._ChannelCoef)
            assert(OutputChannelNum>=1)
            Filter=tf.get_variable(name="conv_filter",shape=[Conv2D._Height,Conv2D._Width,Shape[3],OutputChannelNum],dtype=tf.float32)
            self.Tensor=tf.nn.conv2d(   input=InputTensor,
                                        filter=Filter,
                                        strides=[1,Conv2D._Stride,Conv2D._Stride,1],
                                        padding='SAME',
                                        data_format=Data_Format)
            OutputShape=self.Tensor.get_shape().as_list()
            self.SetImageAttr(*OutputShape[1:])
            
        def CheckValid(self,InputList):
            InputOp=InputList[0]
            InputTensor=InputOp.GetTensor()
            #print(InputTensor)
            Shape=InputTensor.get_shape().as_list()
            print("Check Valid",Shape)
            OutputChannelNum=int(Shape[3]*Conv2D._ChannelCoef)
            if OutputChannelNum>0:
                return True
            else:
                return False
    return Conv2D
    
def PoolingFactory(Size,Stride,Type):
    assert Type in ['Max','Avg']
    class Pooling(ImageOperator):
        _Size=Size
        _Stride=Stride
        _Type=Type
        InputDegree=1
        OutputDegree=1        
        Name='Pooling'
        def __init__(self,InputList,ID,BuildFlag=True):
            super(Pooling,self).__init__(  InputList=InputList,ID=ID,BuildFlag=BuildFlag )
        def ConstructFunc(self,InputList):
            InputOp=InputList[0]
            InputTensor=InputOp.GetTensor()
            if Pooling.Type=='Max':
                self.Tensor=tf.nn.max_pool( InputTensor,
                                                Size=Pooling._Size,
                                                strides=Pooling._Stride,
                                                padding='SAME',
                                                data_format=Data_Format)
            elif Pooling.Type=='Avg':
                self.Tensor=tf.nn.avg_pool( InputTensor,
                                                Size=Pooling._Size,
                                                strides=Pooling._Stride,
                                                padding='SAME',
                                                data_format=Data_Format)

        
    return Pooling
    
def TransConv2DFactory(Size,ImageCoef,ChannelCoef,Stride):
    class TransConv2D(ImageOperator):
        _ImageCoef=ImageCoef
        _Stride=Stride
        _Height=Size
        _Width=Size
        _ChannelCoef=ChannelCoef
        Name='TransConv2D'
        def __init__(self,InputList,ID,BuildFlag=True):
            super(TransConv2D,self).__init__(InputList=InputList,ID=ID,BuildFlag=BuildFlag)
        def ConstructFunc(self,InputList):
            InputOp=InputList[0]
            InputTensor=InputOp.GetTensor()
            Shape=InputTensor.get_shape().as_list()
            OutputChannelNum=int(Shape[3]*TransConv2D._ChannelCoef)
            OutputHeight=int(Shape[1]*TransConv2D._ImageCoef)
            OutputWidth=int(Shape[2]*TransConv2D._ImageCoef)            
            #[height, width, output_channels, in_channels]
            Filter=tf.get_variable(name="trans_conv_filter",shape=[TransConv2D._Height,TransConv2D._Width,OutputChannelNum,Shape[3]],dtype=tf.float32) #"""check"""
            OutputShape=[Shape[0],OutputHeight,OutputWidth,OutputChannelNum]
            self.Tensor=tf.nn.conv2d_transpose( value=InputTensor,
                                                filter=Filter,
                                                output_shape=OutputShape,
                                                strides=TransConv2D._Stride,
                                                padding='SAME',
                                                data_format=Data_Format)
            OutputShape=self.Tensor.get_shape().as_list()
            self.SetImageAttr(*OutputShape[1:])
            
        def CheckValid(self,InputList):
            InputOp=InputList[0]
            InputTensor=InputOp.GetTensor()
            #print(InputTensor)
            Shape=InputTensor.get_shape().as_list()
            #print(Shape)
            OutputChannelNum=int(Shape[3]*TransConv2D._ChannelCoef)
            if OutputChannelNum>0:
                return True
            else:
                return False
                
    return TransConv2D  
    
def ActivationFactory(Type):
    class Activation(ImageOperator):
        _Type=Type
        InputDegree=1
        OutputDegree=1        
        Name='Activation'
        def __init__(self,InputList,ID,BuildFlag=True):
            super(Activation,self).__init__(InputList=InputList,ID=ID)
        def ConstructFunc(self,InputList):
            InputOp=InputList[0]
            InputTensor=InputOp.GetTensor()
            if Activation._Type=='Relu':
                self.Tensor=tf.nn.relu(features=InputTensor)
            elif  Activation._Type=='Leaky_Relu':
                self.Tensor=tf.nn.leaky_relu(features=InputTensor)
            elif  Activation._Type=='L2_Norm':
                self.Tensor=tf.nn.l2_normalize(features=InputTensor)
    return Activation

def BinaryOpFactory(Type):
    class BinaryOp(ImageOperator):
        _Type=Type
        InputDegree=2
        OutputDegree=1
        Name='BinaryOp'
        def __init__(self,InputList,ID,BuildFlag=True):
            super(BinaryOp,self).__init__(InputList=InputList,ID=IDS)
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
        InputDegree=1
        OutputDegree=OutputNum
        Name='Reuse'
        def __init__(self,InputList,ID,BuildFlag=True):
            super(Reuse,self).__init__(InputList=InputList,ID=ID,BuildFlag=BuildFlag)
        def ConstructFunc(self,InputList):
            self.Tensor=InputList[0]
    return Reuse
            
    
def ConcatFactory(InputNum):
    class Concat(ImageOperator):
        _InputNum=InputNum
        InputDegree=InputNum
        OutputDegree=1
        Name='Concat'
        def __init__(self,InputList,ID,BuildFlag=True):
            super(Concat,self).__init__(InputList=InputList,ID=ID,BuildFlag=BuildFlag)
        def ConstructFunc(self,InputList):
            self.Tensor=ConcatOperator(InputList)
            OutputShape=self.Tensor.get_shape().as_list()
            self.SetImageAttr(*OutputShape[1:])            
    return Concat
    
def DenseFactory(HiddenNumCoef):
    class Dense(ImageOperator):
        _HiddenNumCoef=HiddenNumCoef
        InputDegree=1
        OutputDegree=1        
        Name='Dense'
        def __init__(self,InputList,ID,BuildFlag=True):
            super(Dense,self).__init__(InputList=InputList,ID=ID,BuildFlag=BuildFlag)
        def ConstructFunc(self,InputList):
            Input=InputList[0]
            InputTensor=Input.GetTensor()
            InputTensor=tf.reshape(InputTensor,[BatchSize,-1])
            Height,Width,Channel=Input.GetImageAttr()
            Height=int(HiddenNumCoef*Height)
            Width=int(HiddenNumCoef*Width)
            
            self.SetImageAttr(Height,Width,Channel)
            self.Tensor=tf.layers.Dense(input=InputTensor,units=Height*Width*Channel)
    
    return Dense