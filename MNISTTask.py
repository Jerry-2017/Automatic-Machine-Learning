import tensorflow as tf
import logging
from mnist import MNIST
import datetime
from Graph import Graph
from QLearning import QLearning
from ImageOperators import *#ConcatOperator,ImageInput,Conv2DFactory,PoolingFactory,TransConv2DFactory,ActivationFactory,BinaryOpFactory,ReuseFactory,DenseFactory,ConcatOperatorDense
import numpy as np
import os

mndata = MNIST('./')
mndata.gz = True
images, labels = mndata.load_training()
images = np.array(images,dtype=np.float32)
images=images.reshape([-1,28,28,1])
labels= np.array(labels,dtype=np.int32)
print ("Input Shape",images.shape,labels.shape)

###### Experiment Attributes
OperatorLimit=25
BatchSize=64
OperatorSupport=[]
MNIST_IMAGE_WIDTH=28
MNIST_IMAGE_HEIGHT=28
TrainEpochs=10000

######


Op_List=[]

#Convolution
Op_List.append(Conv2DFactory(Size=4,ChannelCoef=2,Stride=1))
Op_List.append(Conv2DFactory(Size=3,ChannelCoef=1,Stride=2))
#Op_List.append(Conv2DFactory(Size=2,ChannelCoef=0.5,Stride=1))
Op_List.append(Conv2DFactory(Size=2,ChannelCoef=1,Stride=1))
#Op_List.append(Conv2DFactory(Size=2,ChannelCoef=0.5,Stride=2))

#Trans Convolution
#Op_List.append(TransConv2DFactory(Size=3,ChannelCoef=2,Stride=1))
#Op_List.append(TransConv2DFactory(Size=3,ChannelCoef=0.5,Stride=1))
#Op_List.append(TransConv2DFactory(Size=2,ChannelCoef=2,Stride=2,ImageCoef=2))
#Op_List.append(TransConv2DFactory(Size=2,ChannelCoef=0.5,Stride=2,ImageCoef=2))

#Dense
Op_List.append(DenseFactory(HiddenNumCoef=2))
Op_List.append(DenseFactory(HiddenNumCoef=1))
Op_List.append(DenseFactory(HiddenNumCoef=0.5))

#Reuse
Op_List.append(ReuseFactory(OutputNum=2))

#Binary_Op
Op_List.append(BinaryOpFactory(Type='Concat'))
Op_List.append(BinaryOpFactory(Type='Add'))

#Pooling
Op_List.append(PoolingFactory(Size=2,Stride=2,Type='Max'))
Op_List.append(PoolingFactory(Size=2,Stride=2,Type='Avg'))

#Activation
Op_List.append(ActivationFactory(Type='Relu'))
Op_List.append(ActivationFactory(Type='Tanh'))


def NetworkDecor(Input,Labels):
    if Input.shape.as_list()[1:]!=get_size_except_dim(Input):
        Reshape=tf.reshape(Input,shape=[BatchSize,get_size_except_dim(Input)])
    Output=tf.layers.dense(inputs=Reshape,units=10,activation=None)
    Labels=tf.cast(tf.reshape(Labels,shape=[BatchSize]),tf.int64)
    #OneHotLabels=tf.one_hot(Labels,depth=10,axis=-1)
    Loss=tf.losses.sparse_softmax_cross_entropy(labels=Labels,logits=Output)
    Acc=tf.reduce_mean(tf.cast(tf.equal(Labels, tf.argmax(Output,1)),tf.float32))
    #print(Loss,Loss.shape.as_list())
    #exit()
    #Loss=tf.reshape(Loss,shape=[-1,1])
    return Output,Loss,Acc

Mode="Train"

RL_Exp=QLearning()
TaskSpec={  "LogHistory":True,
            "OperatorList":Op_List,
            "OperatorNum":OperatorLimit,
            "InputNum":1,
            "OutputNum":1,
            "TaskInput":images,
            "TaskLabel":labels,
            "Epochs":TrainEpochs,
            "NetworkDecor":NetworkDecor,
            "BatchSize":BatchSize,
            "ConcatOperator":ConcatOperatorDense,
            "InputOperator":ImageInput,
            "TrajectoryLength":OperatorLimit-4,
            "RewardGamma":0.9
            }
            
if Mode=="Train":      


    logging.getLogger().setLevel(logging.DEBUG)
    now = datetime.datetime.now()
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='%s_%s.log'%("MNISTTask",now.strftime("%Y-%m-%d %H-%M")),
                    filemode='w')
                    
    RL_Exp.StartTrial(TaskSpec)
elif Mode=="TestBuildGraph":
    OptionList=[[1, 1, 1, 0],[11, 1, 1, 1],[13,1,1,2],[2, 1, 1, 3],[12, 1, 1, 4],[13,1,1,5],[6, 1, 1, 6]]
    RL_Exp.DebugTrainNet(TaskSpec,OptionList)
elif Mode=="TestUnifiedTrans":
    g=Graph(10,Op_List,1,1,ConcatOperatorDense,InputOperator=ImageInput)
    OptionList=[[1, 1, 1, 0],[11, 1, 1, 1],[13,1,1,2],[2, 1, 1, 3],[12, 1, 1, 4],[13,1,1,5],[6, 1, 1, 6]]
    np.set_printoptions(threshold=np.inf)
    for Option in OptionList:
        g.ApplyOption(Option)
        print(np.array(g.UnifiedTransform("3D_NoNull").astype(np.int8)))