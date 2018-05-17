import tensorflow as tf

from ImageOperators import *
import numpy as np
from Graph import Graph,Operator
Data_Format='NHWC'

class QLearning():
    def __init__(self):
        self.sess=tf.Session()
        self.TaskDataInit=False
        pass
        
    def ConstructQFunc3D(self,ImageSize=5,BitDepth=7,BatchSize=5):
        #print("BITDEPTH",BitDepth)
        self.QNetData = tf.placeholder(tf.float32 , shape=[None,BitDepth,ImageSize,ImageSize,1],name="QNet_input")
        self.QNetLabel = tf.placeholder(tf.float32, shape=[None,1], name="QNet_label" )
        
        DatasetRaw = tf.data.Dataset.from_tensor_slices((self.QNetData,self.QNetLabel))
        Dataset=DatasetRaw.repeat().batch(BatchSize)
        self.DataIter = Dataset.make_initializable_iterator()      
        NextData,NextLabel = self.DataIter.get_next()    
        
        #Conv3D_Layer1_Kernel=tf.Variable(tf.initializers.random_normal(shape=[BitDepth,2,2,1,3]),name="Conv3D_Layer1_Kernel",dtype=tf.float32,)
        #[filter_depth, filter_height, filter_width, in_channels, out_channels]
        Conv3D_Layer1=tf.layers.conv3d( inputs=NextData,
                                    filters=3,
                                    kernel_size=(BitDepth,2,2),
                                    strides=[BitDepth,1,1],
                                    padding="SAME")
        print("Conv3D Layer1 shape",Conv3D_Layer1.shape)
        _shape=Conv3D_Layer1.shape
        _shape=[-1,*_shape[2:]]
        Reshape3D2D=tf.reshape(Conv3D_Layer1,shape=_shape)
        #print(Reshape3D2D.shape)
        Pooling2D_Layer1=tf.nn.max_pool(value=Reshape3D2D,
                                        ksize=[1,2,2,1],
                                        strides=[1,1,1,1],
                                        padding="SAME")
                                        
        #Conv2D_Layer2_Kernel=tf.Variable("Conv2D_Layer2_Kernel",shape=[2,2,3,6],dtype=tf.float32,initializer=tf.initializers.random_normal)
        Conv2D_Layer2=tf.layers.conv2d(inputs=Pooling2D_Layer1,
                                        filters=6,
                                        kernel_size=[2,2],
                                        strides=[1,1],
                                        padding="SAME")
        
        Pooling2D_Layer2=tf.nn.max_pool(value=Conv2D_Layer2,
                                        ksize=[1,2,2,1],
                                        strides=[1,1,1,1],
                                        padding="SAME")

        #Conv2D_Layer3_Kernel=tf.Variable("Conv2D_Layer3_Kernel",shape=[2,2,6,12],dtype=tf.float32,initializer=tf.initializers.random_normal)
        Conv2D_Layer3=tf.layers.conv2d(inputs=Pooling2D_Layer2,
                                        kernel_size=[2,2],
                                        filters=12,
                                        strides=[1,1],
                                        padding="SAME")
        
        Pooling2D_Layer3=tf.nn.max_pool(value=Conv2D_Layer3,
                                        ksize=[1,2,2,1],
                                        strides=[1,1,1,1],
                                        padding="SAME")                                      
        _shape=Pooling2D_Layer3.shape
                                        
        Reshape_Layer3=tf.reshape(Pooling2D_Layer3,shape=[-1,_shape[1]*_shape[2]*_shape[3]])
        #print(Reshape_Layer3.shape)
        Dense_Layer4=tf.layers.dense(inputs=Reshape_Layer3, units=256, activation=tf.nn.relu)
        Dropout_Layer4=tf.layers.dropout(inputs=Dense_Layer4,rate=0.5)
        
        Output_Layer=tf.layers.dense(inputs=Reshape_Layer3, units=1, activation=tf.nn.sigmoid)

        self.QNetOutput=Output_Layer
        
        self.QNetLoss = tf.losses.mean_squared_error(labels=NextLabel, predictions=self.QNetOutput)
        
        self.Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        self.QNetTrain_op = self.Optimizer.minimize(
                loss=self.QNetLoss,
                global_step=tf.train.get_global_step())           
    
    def SetNetworkGenerator(self,NetworkGenerator):
        self.NetworkGenerator=NetworkGenerator
    
    def SetOperatorList(self,OperatorList):
        self.OperatorList=OperatorList
    
    def BuildTaskGraph(self,Graph,OutputDecor,ID):
        with tf.variable_scope("TaskNet") as scope:
            Temp_Output=Graph.BuildGraph([self.TaskNextData],ScopeID=ID)
        self.sess.run( self.TaskDataIter.initializer)#,feed_dict={self.TaskNetData:self.TaskNextData,self.TaskNetLabel:self.TaskNextLabel})
        self.TaskOutput,self.TaskLoss=OutputDecor(Temp_Output,self.TaskNextLabel)
        self.TaskTrain = self.Optimizer.minimize(
                loss=self.TaskLoss,
                global_step=tf.train.get_global_step())        ## Need to Change
        
    
    def InitializeTaskGraph(self,Data,Label,BatchSize):
        #print(Data.shape)
        self.TaskNetData = tf.placeholder(tf.float32 , shape=[None,*Data.shape[1:]],name="TaskNet_input")
        self.TaskNetLabel = tf.placeholder(tf.float32, shape=[None,*Label.shape[1:]], name="TaskNet_label" )
        DatasetRaw = tf.data.Dataset.from_tensor_slices((Data,Label))
        Dataset=DatasetRaw.repeat().batch(BatchSize)
        self.TaskDataIter = Dataset.make_initializable_iterator()      
        self.TaskNextData,self.TaskNextLabel = self.TaskDataIter.get_next()     
        self.TaskDataInit=True    
        print("TaskDataIter shape",self.TaskNextData.shape)
        
    def StartTrial(self,TaskSpec):
        LogHistory=TaskSpec["LogHistory"]
        OperatorList=TaskSpec["OperatorList"]
        NetworkDecor=TaskSpec["NetworkDecor"]
        VertexNum=TaskSpec["OperatorNum"]
        InputNum=TaskSpec["InputNum"]
        OutputNum=TaskSpec["OutputNum"]
        BatchSize=TaskSpec["BatchSize"]
        Epochs=TaskSpec["Epochs"]
        ConcatOperator=TaskSpec["ConcatOperator"]
        InputOperator=TaskSpec["InputOperator"]
        TrajectoryLength=TaskSpec["TrajectoryLength"]
        
        TaskInput=TaskSpec["TaskInput"]
        TaskLabel=TaskSpec["TaskLabel"]
        self.HisNet=[]
        self.HisNetPerf=[]
        
        #ImageOperators
        SetBatchSize(BatchSize)
        self.ConstructQFunc3D(ImageSize=VertexNum,BitDepth=len(OperatorList),BatchSize=BatchSize)
        self.InitializeTaskGraph(Data=TaskInput,Label=TaskLabel,BatchSize=BatchSize)
        self.sess.run(tf.global_variables_initializer())
        
        QNet_Format="3D_NoNull"
        for i in range(Epochs):
            Gph=Graph(  VertexNum=VertexNum,
                        OperatorList=OperatorList,
                        InputNum=InputNum,
                        OutputNum=OutputNum,
                        ConcatOperator=ConcatOperator,
                        InputOperator=InputOperator
                        )
            Gph.InitializeCheckOptionInput([self.TaskNextData])
            for j in range(TrajectoryLength):
                OptionList=Gph.ConnectOptions()
                ValidOptionList=[]
                
                for Option in OptionList:
                    if Gph.CheckOption(Option):
                        ValidOptionList.append(Option)
                QNetInputList=[]
                for Option in ValidOptionList:   
                    Gph.ApplyOption(Option)
                    QNetInput=Gph.UnifiedTransform(QNet_Format)
                    QNetInputList.append(QNetInput)
                    Gph.RevokeOption(Option)
                    
                QNetInput=np.array(QNetInputList)
                
                print("QNetInput.shape",QNetInput.shape)
                self.sess.run(self.DataIter.initializer,feed_dict={self.QNetData:QNetInput,self.QNetLabel:np.zeros([QNetInput.shape[0],1])})
                QValuesAll=self.sess.run(self.QNetOutput)
                QValuesClip=QValuesAll[:len(OptionList)]
                #print("All shape",QValuesAll.shape,"Clip ",QValuesClip.shape)
                PossQValues=np.exp(QValuesClip)
                _Sum=np.sum(PossQValues)
                ExpDist=PossQValues/_Sum
                ChosenOption=self.MakeChoice(Distribution=ExpDist,ChoiceList=ValidOptionList)
                
                Gph.ApplyOption(ChosenOption)
                
                Step=100
                #print(i,j)
                self.BuildTaskGraph(Graph=Gph,OutputDecor=NetworkDecor,ID=i)
                Performance=self.TrainTaskNet()
                
                HisItem={"OptionList":Gph.ConnectOptions(),"TrainStep":Step,"Performance":Performance,"UnifiedNet":Gph.UnifiedTransform(QNet_Format)}
                if LogHistory==True:
                    self.Log(HisItem)
                self.HisNet.append(HisItem["UnifiedNet"])
                self.HisNetPerf.append(HisItem["Performance"])
            QStep=100
            self.TrainQNet(self.QNetOutput,self.HisNet,self.HisNetPerf,QStep)
    
    def MakeChoice(self,Distribution,ChoiceList=None):
        RandPoss=np.random.random()
        assert abs(np.sum(Distribution)-1)<1e-5
        AccuPoss=0
        
        for i in range(len(Distribution)):
            AccuPoss+=Distribution[i]
            if RandPoss<=AccuPoss:
                choice=i
                break
        
        if ChoiceList is not None:
            return ChoiceList[choice]
        else:
            return choice
    
    def Log(self,Log):
        pass
        #print(Log)
                
    def TrainQNet(self,Output,Data,Label,Step):
    
        sess=self.sess
        Data=np.array(Data)
        Label=np.array(Label)
        print("DATA",Data.shape,"LABEL",Label.shape)
        
        for i in range(Step):                
            _,acc=sess.run([self.QNetTrain_op,self.QNetLoss],feed_dict={self.QNetData:Data,self.QNetLabel:Label})
            #print(acc)
                
    def TrainTaskNet(self,Step=100):
        sess=self.sess
        sess.run(tf.global_variables_initializer())
        for i in range(Step):                
            _,acc=sess.run([self.TaskTrain,self.TaskLoss])
            #print(acc)
        return [acc]
   