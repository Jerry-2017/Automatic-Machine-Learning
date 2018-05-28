import tensorflow as tf
from Graph import Operator

Data_Format='NHWC'

OUTPUT_CHANNEL_MAX=128
OUTPUT_IMAGE_WIDTH_MAX=60
OUTPUT_IMAGE_HEIGHT_MAX=60
MAX_DENSE_CONNECTION=500
PADDING_TYPE="SAME"

def get_size_except_dim(Tensor,dim=0):
    TensorShape=Tensor.get_shape().as_list()
    _dim=0
    size=1
    for i in TensorShape:
        if _dim!=dim:
            size*=i
        _dim+=1
    return size    

def get_size_except_dims(Tensor,dim=[0]):
    TensorShape=Tensor.get_shape().as_list()
    _dim=0
    size=1
    for i in TensorShape:
        if _dim not in dim:
            size*=i
        _dim+=1
    return size        
    
def SetBatchSize(Batch_Size):
    global BatchSize
    BatchSize=Batch_Size

def ConcatImageTensor(TensorList):
    LenList=len(TensorList)
    #BatchSize=TensorList[0].get_shape().as_list()[0]
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
        Paddings=[[0,0],[0,MaxHeight-TensorShape[1]],[0,MaxWidth-TensorShape[2]],[0,0]]
        PadTensor.append(tf.pad(TensorList[i],Paddings,'CONSTANT'))
    ConcatTensor=tf.concat(PadTensor,axis=3)
    return ConcatTensor

def ConcatOperatorDense(OperatorList):
    TensorList=[]
    TensorList=[t.GetTensor() for t in OperatorList]
    TensorReshape=[tf.reshape(t,[-1,get_size_except_dim(t)]) for t in TensorList]
    result=tf.concat(values=TensorReshape,axis=1)
    print("Concat Shape",result.get_shape().as_list())
    return result
    
def ConcatOperator(OperatorList):
    TensorList=[]
    #print(OperatorList)
    for Input in OperatorList:
        #print(Input)
        Tensor=Input.GetTensor()
        TensorShape=Tensor.get_shape().as_list()
        H,W,C=Input.GetImageAttr()
        if TensorShape==(None,H,W,C):
            ReshapeTensor=Tensor
        else:
            ReshapeTensor=tf.reshape(Tensor,[BatchSize,H,W,C])
        TensorList.append(ReshapeTensor)
    Tensor=ConcatImageTensor(TensorList)
    return Tensor


class ImageOperator(Operator):
    def __init__(self,InputList,ID,BuildFlag=True):
        if len(InputList)==1 and isinstance(InputList[0],ImageOperator):
            self.SetImageAttr(*InputList[0].GetImageAttr())
        super(ImageOperator,self).__init__(InputList=InputList,ID=ID,BuildFlag=BuildFlag)
        if self.Tensor is not None:
            TS=self.Tensor.get_shape().as_list()
            if len(TS)==4:
                self.SetImageAttr(TS[1],TS[2],TS[3])
            
    
    def SetImageAttr(self,Height,Width,Channel):
        self.Width=Width
        self.Height=Height
        self.Channel=Channel
    def GetImageAttr(self):
        return self.Width,self.Height,self.Channel

    def CheckValid(self,InputList):
        return True

    def RestoreShape(self,Tensor):
        if Tensor.get_shape().as_list()!=[None,self.Height,self.Width,self.Channel]:
            return tf.reshape(Tensor,shape=[-1,self.Height,self.Width,self.Channel])
        else:
            return Tensor
        
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
            print(Conv2D._ChannelCoef)
            OutputChannelNum=int(Shape[3]*Conv2D._ChannelCoef)
            assert(OutputChannelNum>=1)
            init=tf.initializers.random_normal()
            self.Tensor=tf.layers.conv2d(  inputs=InputTensor,
                                        kernel_size=(Conv2D._Height,Conv2D._Width),
                                        filters=OutputChannelNum,
                                        strides=(Conv2D._Stride,Conv2D._Stride),
                                        padding=PADDING_TYPE,
                                        data_format="channels_last",
                                        name="conv2d")
            OutputShape=self.Tensor.get_shape().as_list()
            self.SetImageAttr(*OutputShape[1:])
            
        def CheckValid(self,InputList):
            InputOp=InputList[0]
            InputTensor=InputOp.GetTensor()
            #print(InputTensor)
            Shape=InputTensor.get_shape().as_list()
            #print("Check SAME",Shape)
            OutputChannelNum=int(Shape[3]*Conv2D._ChannelCoef)
            if OutputChannelNum>0  and OutputChannelNum<OUTPUT_CHANNEL_MAX and  Conv2D._Stride<Shape[1] and  Conv2D._Stride<Shape[2]:
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
            if Pooling._Type=='Max':
                self.Tensor=tf.layers.max_pooling2d( inputs=InputTensor,
                                                pool_size=(Pooling._Size,Pooling._Size),
                                                strides=(Pooling._Stride,Pooling._Stride),
                                                padding=PADDING_TYPE,
                                                data_format='channels_last')
            elif Pooling._Type=='Avg':
                self.Tensor=tf.layers.average_pooling2d( inputs=InputTensor,
                                                pool_size=(Pooling._Size,Pooling._Size),
                                                strides=(Pooling._Stride,Pooling._Stride),
                                                padding=PADDING_TYPE,
                                                data_format='channels_last')

        
    return Pooling
    
def TransConv2DFactory(Size,ChannelCoef,Stride):
    class TransConv2D(ImageOperator):

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
            #OutputHeight=int(Shape[1]*TransConv2D._ImageCoef)
            #OutputWidth=int(Shape[2]*TransConv2D._ImageCoef)            
            #[height, width, output_channels, in_channels]
            #Filter=tf.get_variable(name="trans_conv_filter",shape=[TransConv2D._Height,TransConv2D._Width,OutputChannelNum,Shape[3]],dtype=tf.float32) #"""check"""
            #OutputShape=[BatchSize,OutputHeight,OutputWidth,OutputChannelNum]
            #self.Tensor=tf.nn.conv2d_transpose( value=InputTensor,
            #                                    filter=Filter,
            #                                    output_shape=OutputShape,
            #                                    strides=[1,TransConv2D._Stride,TransConv2D._Stride,1],
            #                                    padding='SAME',
            #                                    data_format=Data_Format)
            self.Tensor=tf.layers.conv2d_transpose( inputs=InputTensor,
                                                filters=OutputChannelNum,
                                                kernel_size=(TransConv2D._Height,TransConv2D._Width),
                                                strides=[TransConv2D._Stride,TransConv2D._Stride],
                                                padding=PADDING_TYPE,
                                                data_format="channels_last",
                                                name="trans")
            OutputShape=self.Tensor.get_shape().as_list()
            self.SetImageAttr(*OutputShape[1:])
            
        def CheckValid(self,InputList):
            InputOp=InputList[0]
            InputTensor=InputOp.GetTensor()
            #print(InputTensor)
            Shape=InputTensor.get_shape().as_list()
            #print(Shape)
            OutputChannelNum=int(Shape[3]*TransConv2D._ChannelCoef)
            if OutputChannelNum>0 and OutputChannelNum<OUTPUT_CHANNEL_MAX and InputOp.Width<=OUTPUT_IMAGE_WIDTH_MAX and InputOp.Height<=OUTPUT_IMAGE_HEIGHT_MAX:
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
            super(Activation,self).__init__(InputList=InputList,ID=ID,BuildFlag=BuildFlag)
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
            super(BinaryOp,self).__init__(InputList=InputList,ID=ID,BuildFlag=BuildFlag)
        def ConstructFunc(self,InputList):
            InputOp1=InputList[0]
            InputOp2=InputList[1]
            InputOp1Tensor=InputOp1.GetTensor()
            InputOp2Tensor=InputOp2.GetTensor()
            if BinaryOp._Type=='Concat':
                self.Tensor=ConcatImageTensor([InputOp1Tensor,InputOp2Tensor])
            elif BinaryOp._Type=='Add':
                assert InputOp1Tensor.shape.as_list()[1:]==InputOp2Tensor.shape.as_list()[1:]
                self.Tensor=tf.add(InputOp1Tensor,InputOp2Tensor)
            
        def CheckValid(self,InputList):
            InputOp1=InputList[0]
            InputOp2=InputList[1]
            InputOp1Tensor=InputOp1.GetTensor()
            InputOp2Tensor=InputOp2.GetTensor()
            #print("cmp",InputOp1Tensor.shape.as_list(),InputOp2Tensor.shape.as_list())
            if BinaryOp._Type=='Add':                 
                if InputOp1Tensor.shape.as_list()[1:]!=InputOp2Tensor.shape.as_list()[1:]:
                    return False
            return True
                       
    return BinaryOp

def ReuseFactory(OutputNum):
    class Reuse(ImageOperator):
        _OutputNum=OutputNum
        InputDegree=1
        OutputDegree=OutputNum
        Name='Reuse'
        def __init__(self,InputList,ID,BuildFlag=True):
            super(Reuse,self).__init__(InputList=InputList,ID=ID,BuildFlag=BuildFlag)
        def ConstructFunc(self,InputList):
            self.Tensor=InputList[0].GetTensor()
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
            #BatchSize=InputTensor.get_shape().as_list()[0]            
            Height,Width,Channel=InputTensor.shape.as_list()[1:]
            InputTensor=tf.reshape(InputTensor,[BatchSize,get_size_except_dims(InputTensor,dim=[0])])
            Height=int(HiddenNumCoef*Height)
            Width=int(HiddenNumCoef*Width)
            
            self.SetImageAttr(Height,Width,Channel)            
            self.Tensor=tf.layers.dense(inputs=InputTensor,units=Height*Width*Channel,activation=None,name="dense")
            self.Tensor=self.RestoreShape(self.Tensor)            
        def CheckValid(self,InputList):
            Input=InputList[0]
            InputTensor=Input.GetTensor()
            dim=get_size_except_dim(InputTensor)
            Height,Width,Channel=Input.GetImageAttr()
            Height=int(HiddenNumCoef*Height)
            Width=int(HiddenNumCoef*Width)            
            if Height*Width*Channel<1 or dim*HiddenNumCoef*HiddenNumCoef>MAX_DENSE_CONNECTION:
                return False
            return True
    return Dense