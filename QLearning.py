import tensorflow as tf
import logging

from ImageOperators import *
import numpy as np
from Graph import Graph,Operator
Data_Format='NHWC'


class EarlyStop():
    def __init__(self,GlideWindow=200,FailMax=100):
        self.GlideWindow=GlideWindow
        self.FailMax=FailMax
        self.Reset()
    def Reset(self):
        self.Loss=[]
        self.TrajCount=0
        self.FailCnt=0
    def AddLoss(self,Loss):
        self.TrajCount+=1
        self.Loss.append(Loss)
    def ShouldStop(self):
        _Begin=max(0,self.TrajCount-self.GlideWindow)
        _Stop=self.TrajCount
        _Good=True
        if self.Loss[-1]<1e-5:
            return True
        for i in range(_Begin ,_Stop):
            if self.Loss[i]<self.Loss[-1]:
                _Good=False
                break
        if not _Good:
            self.FailCnt+=1
        else:
            self.FailCnt=0
        if self.FailCnt>self.FailMax:
            return True
        else:
            return False
        
        
        

class QLearning():
    def __init__(self):
        self.QGraph=tf.Graph()
        self.QNetsess=tf.Session(graph=self.QGraph)
        self.Optimizer = tf.train.AdamOptimizer(1e-4)
        self.TaskDataInit=False
        pass
        
    def ConstructQFunc3D(self,ImageSize=5,BitDepth=7,BatchSize=5):
        #print("BITDEPTH",BitDepth)
        with tf.variable_scope("QNet") as scope:
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
            
            self.QNetTrain_op = self.Optimizer.minimize(
                    loss=self.QNetLoss)
                #,global_step=tf.train.get_global_step())           
    
    def SetNetworkGenerator(self,NetworkGenerator):
        self.NetworkGenerator=NetworkGenerator
    
    def SetOperatorList(self,OperatorList):
        self.OperatorList=OperatorList
    
    def BuildTaskGraph(self,Graph,OutputDecor,ID,CheckNodeShape=True):
        with tf.variable_scope("TaskNet") as scope:
            Temp_Output=Graph.BuildGraph([self.TaskNextData],ScopeID=ID)
        self.TaskOutput,self.TaskLoss,self.TaskAcc=OutputDecor(Temp_Output,self.TaskNextLabel)
        Rst=Graph.StrOptionList()
        self.Log("Current Net Arc %s"%Rst)
        print("Temp_Output",Temp_Output.shape)
        self.TaskSess.run( self.TaskDataIter.initializer)#,feed_dict={self.TaskNetData:self.TaskNextData,self.TaskNetLabel:self.TaskNextLabel})
        if CheckNodeShape:
            self.Op_Shape=Graph.GetGraphNodeShape()

        self.TaskTrain = self.Optimizer.minimize(
                loss=self.TaskLoss,
                global_step=tf.train.get_global_step())        ## Need to Change
        
    
    def InitializeTaskGraph(self,Data,Label,BatchSize):
        #print(Data.shape)
        with tf.variable_scope("TaskNet") as scope:
            self.TaskNetData = tf.placeholder(tf.float32 , shape=[None,*Data.shape[1:]],name="TaskNet_input")
            self.TaskNetLabel = tf.placeholder(tf.float32, shape=[None,*Label.shape[1:]], name="TaskNet_label" )
            DatasetRaw = tf.data.Dataset.from_tensor_slices((Data,Label))
            Dataset=DatasetRaw.repeat().batch(BatchSize)
            self.TaskDataIter = Dataset.make_initializable_iterator()      
            self.TaskNextData,self.TaskNextLabel = self.TaskDataIter.get_next()     
            self.TaskDataInit=True    
        print("TaskDataIter shape",self.TaskNextData.shape)

    def DebugTrainNet(self,TaskSpec,OptionList):
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
        RewardGamma=TaskSpec["RewardGamma"]
        
        TaskInput=TaskSpec["TaskInput"]
        TaskLabel=TaskSpec["TaskLabel"]
        
        SetBatchSize(BatchSize)
        
        self.TaskGraph=tf.Graph()
        self.TaskSess=tf.Session(graph=self.TaskGraph)
        
        Gph=Graph(  VertexNum=VertexNum,
                    OperatorList=OperatorList,
                    InputNum=InputNum,
                    OutputNum=OutputNum,
                    ConcatOperator=ConcatOperator,
                    InputOperator=InputOperator
                    )
        with self.TaskGraph.as_default():
            self.InitializeTaskGraph(Data=TaskInput,Label=TaskLabel,BatchSize=BatchSize)     
            for Option in OptionList:
                Gph.ApplyOption(Option)
            self.BuildTaskGraph(Graph=Gph,OutputDecor=NetworkDecor,ID=0)
            for i in Gph.InternalTensor:
                if i is None:
                    continue
                print(i,type(i).Name,i.GetTensor().shape)
            #return    
            Loss,Acc=self.TrainTaskNet(Step=Epochs)
        print(Loss,Acc)
        
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
        RewardGamma=TaskSpec["RewardGamma"]
        
        TaskInput=TaskSpec["TaskInput"]
        TaskLabel=TaskSpec["TaskLabel"]
        self.HisNet=[]
        self.HisNetPerf=[]
        
        self.Log("Operator List "+str(OperatorList))
        
        #ImageOperators
        SetBatchSize(BatchSize)
        
        with self.QGraph.as_default():
            self.ConstructQFunc3D(ImageSize=VertexNum,BitDepth=len(OperatorList),BatchSize=BatchSize)
            #print([i.op_def.name for i in self.QGraph.get_operations()])
            QNetVar=tf.global_variables()#[op for op in tf.GraphKeys.GLOBAL_VARIABLES if isinstance(op,tf.Variable) and  not tf.is_variable_initialized(op)]
            #print([i.name for i in QNetVar])
            self.QNetsess.run(tf.initialize_variables(QNetVar) )#tf.global_variables_initializer())
            
        QNet_Format="3D_NoNull"
        for i in range(Epochs):
            if Epochs % 100==0:
                self.TaskGraph=tf.Graph()
                self.TaskSess=tf.Session(graph=self.TaskGraph)
                with self.TaskGraph.as_default():
                    self.InitializeTaskGraph(Data=TaskInput,Label=TaskLabel,BatchSize=BatchSize)
                
            Gph=Graph(  VertexNum=VertexNum,
                        OperatorList=OperatorList,
                        InputNum=InputNum,
                        OutputNum=OutputNum,
                        ConcatOperator=ConcatOperator,
                        InputOperator=InputOperator
                        )
            Gph.InitializeCheckOptionInput([self.TaskNextData])
            
            TrajectoryStep=0
            HisNetTemp=[]
            HisPerfTemp=[]
            for j in range(TrajectoryLength):
                OptionList=Gph.ConnectOptions()
                ValidOptionList=[]
                
                for Option in OptionList:
                    if Gph.CheckOption(Option):
                        ValidOptionList.append(Option)
                
                
                if ChooseType=='QLearning':        
                    QNetInputList=[]
                    for Option in ValidOptionList:   
                        Gph.ApplyOption(Option)
                        QNetInput=Gph.UnifiedTransform(QNet_Format)
                        QNetInputList.append(QNetInput)
                        Gph.RevokeOption(Option)
                        
                    QNetInput=np.array(QNetInputList)
                    
                    print("QNetInput.shape",QNetInput.shape)
                    self.QNetsess.run(self.DataIter.initializer,feed_dict={self.QNetData:QNetInput,self.QNetLabel:np.zeros([QNetInput.shape[0],1])})
                    QValuesAll=None
                    for _ in range((len(QNetInput)-1)//BatchSize+1):                
                        QValuesPart=self.QNetsess.run(self.QNetOutput)
                        if QValuesAll is None:
                            QValuesAll=QValuesPart
                        else:
                            QValuesAll=np.concatenate((QValuesAll,QValuesPart),axis=0)
                    QValuesClip=QValuesAll[:len(ValidOptionList)]
                    
                    print("All shape",QValuesAll.shape,"Clip ",QValuesClip.shape)
                    PossQValues=np.exp(QValuesClip)
                    _Sum=np.sum(PossQValues)
                    ExpDist=PossQValues/_Sum
                    ChosenOption=self.MakeChoice(Distribution=ExpDist,ChoiceList=ValidOptionList)
                elif ChoosenType=='Random':
                    ChosenOption=OptionList[random.randint(0,len(range(OptionList)))]
                
                self.Log("ChosenOption "+str(ChosenOption))
                Gph.ApplyOption(ChosenOption)
                
                Step=100
                #print(i,j)
                
                with self.TaskGraph.as_default():
                    self.BuildTaskGraph(Graph=Gph,OutputDecor=NetworkDecor,ID=i)
                Loss,Acc=self.TrainTaskNet(Step=Epochs)
                
                HisItem={"Traj":j,"OptionList":Gph.ConnectOptions(),"TrainStep":Step,"Loss":Loss,"Acc":Acc,"UnifiedNet":Gph.UnifiedTransform(QNet_Format)}
                if LogHistory==True:
                    pass
                    #self.Log(HisItem)
                HisNetTemp.append(HisItem["UnifiedNet"])
                HisPerfTemp.append(HisItem["Acc"])
                TrajectoryStep+=1
            self.HisNet.extend(HisNetTemp)
            for i in range(TrajectoryStep-1):
                print("HisNet",HisPerfTemp[i])
                HisPerfTemp[TrajectoryStep-i-2][0]+= RewardGamma*HisPerfTemp[TrajectoryStep-i-1][0]
            self.HisNetPerf.extend(HisPerfTemp)
            QStep=Epochs
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
        logging.info(Log)
        #print(Log)
    
    def TrainQNet(self,Output,Data,Label,Step,EarlyStopFlag=True):
    
        sess=self.QNetsess
        Data=np.array(Data)
        Label=np.array(Label)
        es=EarlyStop()
        print("DATA",Data.shape,"LABEL",Label.shape)
        
        for i in range(Step):                
            if i==0:
                feed_dict={self.QNetData:Data,self.QNetLabel:Label}
            else:
                feed_dict={}
            _,acc=sess.run([self.QNetTrain_op,self.QNetLoss],feed_dict=feed_dict)
            
            if i%100==0:            
                self.Log("QLearning Net Loss %f"%acc)
            
            if EarlyStopFlag:
                es.AddLoss(acc)
                if es.ShouldStop():
                    break
            
            
                
    def TrainTaskNet(self,Step=100,EarlyStopFlag=True):
        
        sess=self.TaskSess
        #[op for op in tf.GraphKeys.GLOBAL_VARIABLES if isinstance(op,tf.Variable) and not tf.is_variable_initialized(op)]
        
        with self.TaskGraph.as_default():
            TaskVar=tf.global_variables()      
            TaskVar+=tf.local_variables()      
            sess.run(tf.initialize_variables(TaskVar) )#tf.global_variables_initializer()) 
        #print([i.name for i in TaskVar])
        OpShape=sess.run(self.Op_Shape)
        self.Log("Operator Shape %s"%(str(OpShape)))
        es=EarlyStop()
        for i in range(Step):                
            _,loss=sess.run([self.TaskTrain,self.TaskLoss])
            
            
            if i%100==0:
                self.Log("Task Net Loss %f"%loss)            
                print(loss)
            if EarlyStopFlag:
                es.AddLoss(loss)
                if es.ShouldStop():
                    break
        EvalEpoch=20
        acc=[]
        for i in range(EvalEpoch):
           Acc=sess.run(self.TaskAcc)
           acc.append(Acc)
        acc=np.sum(acc)/EvalEpoch
        self.Log("Task Net Acc %f"%acc)
        return [loss],[acc]
   