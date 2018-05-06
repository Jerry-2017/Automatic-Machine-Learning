from mnist import MNIST
from QLearning import QLearning
from ImageOperators import ConcatOperator,ImageInput,Conv2DFactory,PoolingFactory,TransConv2DFactory,ActivationFactory,BinaryOpFactory,ReuseFactory,DenseFactory
import numpy as np
mndata = MNIST('./')
mndata.gz = True
images, labels = mndata.load_training()
images = np.array(images,dtype=np.float32)
images=images.reshape([-1,28,28,1])
labels= np.array(labels)
print ("Input Shape",images.shape,labels.shape)

###### Experiment Attributes
OperatorLimit=10
BatchSize=5
OperatorSupport=[]
MNIST_IMAGE_WIDTH=28
MNIST_IMAGE_HEIGHT=28
TrainEpochs=10

######


Op_List=[]

#Convolution
Op_List.append(Conv2DFactory(Size=2,ChannelCoef=2,Stride=1))
Op_List.append(Conv2DFactory(Size=2,ChannelCoef=0.5,Stride=1))
Op_List.append(Conv2DFactory(Size=2,ChannelCoef=2,Stride=2))
Op_List.append(Conv2DFactory(Size=2,ChannelCoef=0.5,Stride=2))

#Trans Convolution
Op_List.append(TransConv2DFactory(Size=2,ChannelCoef=2,Stride=1,ImageCoef=2))
Op_List.append(TransConv2DFactory(Size=2,ChannelCoef=0.5,Stride=1,ImageCoef=2))
Op_List.append(TransConv2DFactory(Size=2,ChannelCoef=2,Stride=2,ImageCoef=2))
Op_List.append(TransConv2DFactory(Size=2,ChannelCoef=0.5,Stride=2,ImageCoef=2))

#Dense
Op_List.append(DenseFactory(HiddenNumCoef=1.5))
Op_List.append(DenseFactory(HiddenNumCoef=0.5))

#Binary_Op
Op_List.append(BinaryOpFactory(Type='Concat'))
Op_List.append(BinaryOpFactory(Type='Add'))

#Pooling
Op_List.append(PoolingFactory(Size=2,Stride=1,Type='Max'))
Op_List.append(PoolingFactory(Size=2,Stride=1,Type='Avg'))

#Activation
Op_List.append(ActivationFactory(Type='Relu'))

def TaskOutput(OutputList):
	Output=OutputList[0]
	OutTensor=Output.GetTensor()
	OutTensorShape=OutTensor.shape()
	
	Reshape=OutTensor.reshape([BatchSize,-1])
	Output=tf.layer.dense(inputs=Reshape,units=10,activation=tf.softmax)
	return Output

def NetworkDecor(Input):
    return Input
    
RL_Exp=QLearning()
TaskSpec={	"LogHistory":True,
            "OperatorList":Op_List,
            "NetworkGenerator":TaskOutput,
            "OperatorNum":OperatorLimit,
            "InputNum":1,
            "OutputNum":1,
            "TaskInput":images,
            "TaskLabel":labels,
            "Epochs":TrainEpochs,
            "NetworkDecor":NetworkDecor,
            "BatchSize":BatchSize,
            "ConcatOperator":ConcatOperator,
            "InputOperator":ImageInput
            }

                
                
RL_Exp.StartTrial(TaskSpec)

