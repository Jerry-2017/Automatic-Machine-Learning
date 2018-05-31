import tensorflow as tf
import logging

from ImageOperators import *
import numpy as np
import random
from Graph import Graph,Operator
Data_Format='NHWC'

def shuffle_union(a,b):
    assert (a.shape[0]==b.shape[0])
    dim=a.shape[0]
    perm=np.random.permutation(dim)
    ra=np.empty(a.shape,a.dtype)
    rb=np.empty(b.shape,b.dtype)
    for old,new in enumerate(perm):
        ra[new]=a[old]
        rb[new]=b[old]
    return ra,rb
        

class DataIter:
    def __init__(self,Data,Label=None,TestProp=0.1,BatchSize=64,InOrder=False):
        self.Data=Data
        self.Label=Label
        self.TestProp=TestProp
        self.BatchSize=BatchSize
        self.InOrder=InOrder
        self.Reset()
        
    def Reset(self):
        if not self.InOrder:
            if self.Label is not None:
                self.Data,self.Label=shuffle_union(self.Data,self.Label)
            else:
                self.Data=np.shuffle(self.Data)
        self.DataNum=self.Data.shape[0]
        self.TestNum=int(self.DataNum*self.TestProp)
        self.TrainNum=self.DataNum-self.TestNum
        self.TestData=self.Data[self.TrainNum:]        
        self.TrainData=self.Data[:self.TrainNum]
        
        if self.Label is not None:
            self.TestLabel=self.Label[self.TrainNum:]
            self.TrainLabel=self.Label[:self.TrainNum]
        self.TestPivot=0
        self.TrainPivot=0
        
    def Get(self,Arr,Start,Size):
        Dim=Arr.shape[0]-1
        RstArr=np.zeros([Size,*Arr.shape[1:]],Arr.dtype)
        PivotRaw=Start
        PivotRst=0
        while Size>0:
            Window=min(Dim-PivotRaw,Size)
            RstArr[PivotRst:PivotRst+Window]=Arr[PivotRaw:PivotRaw+Window]
            PivotRst+=Window
            PivotRaw=(PivotRaw+Window)%Dim
            Size-=Window
        return RstArr,PivotRaw
        
    def NextBatch(self,Test=False):
        if self.Label is not None:
            if Test:
                Data,_=self.Get(self.TestData,self.TestPivot,self.BatchSize)
                Label,self.TestPivot=self.Get(self.TestLabel,self.TestPivot,self.BatchSize)
            else:
                Data,_=self.Get(self.TrainData,self.TrainPivot,self.BatchSize)
                Label,self.TrainPivot=self.Get(self.TrainLabel,self.TrainPivot,self.BatchSize)
            return Data,Label
        else:
            if Test:
                Data,self.TestPivot=self.Get(self.TestData,self.TestPivot,self.BatchSize)
            else:
                Data,self.TrainPivot=self.Get(self.TrainData,self.TrainPivot,self.BatchSize)
            return Data
        
        
        


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
        self.SessConfig = tf.ConfigProto()
        self.SessConfig.gpu_options.allow_growth = True
        self.QGraph=tf.Graph()
        self.QNetsess=tf.Session(graph=self.QGraph,config=self.SessConfig)
        self.Optimizer = tf.train.AdamOptimizer(1e-4)
        self.TaskDataInit=False

        pass
        
    def ConstructQFunc3D(self,ImageSize=5,BitDepth=7,BatchSize=5,OneHotEmbed=None,EmbedLen=3):
        #print("BITDEPTH",BitDepth)
        with tf.variable_scope("QNet") as scope:
            self.QNetData = tf.placeholder(tf.float32 , shape=[None,BitDepth,ImageSize,ImageSize,1],name="QNet_input")
            self.QNetLabel = tf.placeholder(tf.float32, shape=[None,1], name="QNet_label" )
            
            #DatasetRaw = tf.data.Dataset.from_tensor_slices((self.QNetData,self.QNetLabel))
            #Dataset=DatasetRaw.repeat().batch(BatchSize)
            #self.DataIter = Dataset.make_initializable_iterator()      
            #NextData,NextLabel = self.DataIter.get_next()    
            
            NextData,NextLabel=self.QNetData,self.QNetLabel
            
            #Embedding Layer1
            
            _shape=NextData.shape.as_list()
            
            reshape=tf.reshape(NextData,shape=[-1,BitDepth,ImageSize,ImageSize])
            reshape=tf.reshape(tf.transpose(reshape,perm=[0,2,3,1]),shape=[-1,BitDepth])
            Embed=tf.layers.dense(inputs=reshape,units=EmbedLen)
            
            reshape=tf.transpose(tf.reshape(Embed,shape=[-1,*_shape[2:4],EmbedLen]),perm=[0,3,1,2])
            reshape=tf.reshape(reshape,shape=[-1,EmbedLen,ImageSize,ImageSize,1])
            
            
            #Conv3D_Layer1_Kernel=tf.Variable(tf.initializers.random_normal(shape=[BitDepth,2,2,1,3]),name="Conv3D_Layer1_Kernel",dtype=tf.float32,)
            #[filter_depth, filter_height, filter_width, in_channels, out_channels]
            Conv3D_Layer1=tf.layers.conv3d( inputs=reshape,
                                        filters=8,
                                        kernel_size=(EmbedLen,5,5),
                                        strides=[1,1,1],
                                        padding="VALID")
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
                                            filters=8,
                                            kernel_size=[5,5],
                                            strides=[1,1],
                                            padding="SAME")
            
            Pooling2D_Layer2=tf.nn.max_pool(value=Conv2D_Layer2,
                                            ksize=[1,2,2,1],
                                            strides=[1,1,1,1],
                                            padding="SAME")

            #Conv2D_Layer3_Kernel=tf.Variable("Conv2D_Layer3_Kernel",shape=[2,2,6,12],dtype=tf.float32,initializer=tf.initializers.random_normal)
            Conv2D_Layer3=tf.layers.conv2d(inputs=Pooling2D_Layer2,
                                            kernel_size=[5,5],
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
        #self.TaskSess.run( self.TaskDataIter.initializer)#,feed_dict={self.TaskNetData:self.TaskNextData,self.TaskNetLabel:self.TaskNextLabel})
        if CheckNodeShape:
            self.Op_Shape=Graph.GetGraphNodeShape()

        self.TaskTrain = self.Optimizer.minimize(
                loss=self.TaskLoss,
                global_step=tf.train.get_global_step())        ## Need to Change
        
    
    def InitializeTaskGraph(self,Data,Label):
        #print(Data.shape)
        with tf.variable_scope("TaskNet") as scope:
            """
            self.TaskNetData = tf.placeholder(tf.float32 , shape=[None,*Data.shape[1:]],name="TaskNet_input")
            self.TaskNetLabel = tf.placeholder(tf.float32, shape=[None,*Label.shape[1:]], name="TaskNet_label" )

            DatasetRaw = tf.data.Dataset.from_tensor_slices((Data,Label))
            Dataset=DatasetRaw.repeat().batch(BatchSize)
            self.TaskDataIter = Dataset.make_initializable_iterator()      
            self.TaskNextData,self.TaskNextLabel = self.TaskDataIter.get_next()"""
            self.TaskDataIter=DataIter(Data,Label,TestProp=0.1,BatchSize=self.BatchSize)
            self.TaskNextData = tf.placeholder(tf.float32 , shape=[None,*Data.shape[1:]],name="TaskNet_input")
            self.TaskNextLabel = tf.placeholder(tf.float32, shape=[None,*Label.shape[1:]], name="TaskNet_label" )
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
        self.TaskSess=tf.Session(graph=self.TaskGraph,config=self.SessConfig)
        
        Gph=Graph(  VertexNum=VertexNum,
                    OperatorList=OperatorList,
                    InputNum=InputNum,
                    OutputNum=OutputNum,
                    ConcatOperator=ConcatOperator,
                    InputOperator=InputOperator
                    )
        with self.TaskGraph.as_default():
            self.InitializeTaskGraph(Data=TaskInput,Label=TaskLabel)     
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
        self.BatchSize=BatchSize
        
        with self.QGraph.as_default():
            self.ConstructQFunc3D(ImageSize=VertexNum,BitDepth=len(OperatorList),BatchSize=BatchSize)
            #print([i.op_def.name for i in self.QGraph.get_operations()])
            QNetVar=tf.global_variables()#[op for op in tf.GraphKeys.GLOBAL_VARIABLES if isinstance(op,tf.Variable) and  not tf.is_variable_initialized(op)]
            #print([i.name for i in QNetVar])
            self.QNetsess.run(tf.initialize_variables(QNetVar) )#tf.global_variables_initializer())
        
        Initial_Explore_Rate=0.5
        Explore_Rate_Decay=0.001
        QNet_Format="3D_NoNull"
        for i in range(Epochs):
            if Epochs % 100==0:
                self.TaskGraph=tf.Graph()
                self.TaskSess=tf.Session(graph=self.TaskGraph,config=self.SessConfig)
                with self.TaskGraph.as_default():
                    self.InitializeTaskGraph(Data=TaskInput,Label=TaskLabel)
                
            Gph=Graph(  VertexNum=VertexNum,
                        OperatorList=OperatorList,
                        InputNum=InputNum,
                        OutputNum=OutputNum,
                        ConcatOperator=ConcatOperator,
                        InputOperator=InputOperator
                        )
            Gph.InitializeCheckOptionInput([self.TaskNextData])
            if Initial_Explore_Rate>0:
                Initial_Explore_Rate-=Explore_Rate_Decay
            TrajectoryStep=0
            HisNetTemp=[]
            HisPerfTemp=[]
            self.InitializedVar=[]
            for j in range(TrajectoryLength):
                OptionList=Gph.ConnectOptions()
                ValidOptionList=[]
                
                for Option in OptionList:
                    if Gph.CheckOption(Option):
                        ValidOptionList.append(Option)
                
                RandDie=random.random()
                
                if RandDie>Initial_Explore_Rate:
                    ChooseType="QLearning"
                else:
                    ChooseType="Random"
                    
                
                if ChooseType=='QLearning':        
                    QNetInputList=[]
                    for Option in ValidOptionList:   
                        Gph.ApplyOption(Option)
                        QNetInput=Gph.UnifiedTransform(QNet_Format)
                        QNetInputList.append(QNetInput)
                        Gph.RevokeOption(Option)
                        
                    QNetInput=np.array(QNetInputList)
                    
                    QValuesClip=self.PredictQNet(QNetInput)

                    PossQValues=np.exp(QValuesClip)
                    _Sum=np.sum(PossQValues)
                    ExpDist=PossQValues/_Sum
                    ChosenOption=self.MakeChoice(Distribution=ExpDist,ChoiceList=ValidOptionList)
                elif ChooseType=='Random':
                    ChosenOption=ValidOptionList[random.randint(0,len(ValidOptionList)-1)]
                
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
        
    def PredictQNet(self,QNetInput):
        print("QNetInput.shape",QNetInput.shape)
        #self.QNetsess.run(self.DataIter.initializer,feed_dict={self.QNetData:QNetInput,self.QNetLabel:np.zeros([QNetInput.shape[0],1])})
        dataiter=DataIter(QNetInput,BatchSize=self.BatchSize,TestProp=0,InOrder=True)
        QValuesAll=None
        for _ in range((len(QNetInput)-1)//self.BatchSize+1):
            Data=dataiter.NextBatch()
            QValuesPart=self.QNetsess.run(self.QNetOutput,feed_dict={self.QNetData:Data})
            if QValuesAll is None:
                QValuesAll=QValuesPart
            else:
                QValuesAll=np.concatenate((QValuesAll,QValuesPart),axis=0)
        QValuesClip=QValuesAll[:QNetInput.shape[0]]
        
        print("All shape",QValuesAll.shape,"Clip ",QValuesClip.shape)
        return QValuesClip
        
    def TrainQNet(self,Output,Data,Label,Step,EarlyStopFlag=True):        
        sess=self.QNetsess
        Data=np.array(Data)
        Label=np.array(Label)
        self.QNetDataIter=DataIter(Data,Label)
        es=EarlyStop(GlideWindow=20,FailMax=10)
        print("DATA",Data.shape,"LABEL",Label.shape)
        
        for i in range(Step):                
            Data,Label=self.QNetDataIter.NextBatch()
            _,acc=sess.run([self.QNetTrain_op,self.QNetLoss],feed_dict={self.QNetData:Data,self.QNetLabel:Label})
            
            if i%50==0:            
                for i in range(20):
                    Data,Label=self.QNetDataIter.NextBatch(Test=True)
                    loss=sess.run(self.QNetLoss,feed_dict={self.QNetData:Data,self.QNetLabel:Label})
                
                self.Log("QLearning Net Loss %f"%loss)
            
                if EarlyStopFlag:
                    es.AddLoss(loss)
                    if es.ShouldStop():
                        break
            
            
                
    def TrainTaskNet(self,Step=100,EarlyStopFlag=True):
        
        sess=self.TaskSess
        #[op for op in tf.GraphKeys.GLOBAL_VARIABLES if isinstance(op,tf.Variable) and not tf.is_variable_initialized(op)]
        
        with self.TaskGraph.as_default():
            TaskVar=tf.global_variables()      
            TaskVar+=tf.local_variables() 
            NewVar=[]
            for Var in TaskVar:
                if Var.name not in self.InitializedVar:
                    self.InitializedVar.append(Var.name)
                    NewVar.append(Var)
            sess.run(tf.initialize_variables(NewVar) )#tf.global_variables_initializer()) 
        #print([i.name for i in TaskVar])
        Data,Label=self.TaskDataIter.NextBatch()
        OpShape=sess.run(self.Op_Shape,feed_dict={self.TaskNextData:Data,self.TaskNextLabel:Label})
        self.Log("Operator Shape %s"%(str(OpShape)))
        es=EarlyStop(GlideWindow=20,FailMax=10)
        for i in range(Step):                
            Data,Label=self.TaskDataIter.NextBatch()
            #print(Data.shape,Label.shape)
            _,loss=sess.run([self.TaskTrain,self.TaskLoss],feed_dict={self.TaskNextData:Data,self.TaskNextLabel:Label})
            if i%10==0:
                Data,Label=self.TaskDataIter.NextBatch(Test=True)
                acc=sess.run(self.TaskAcc,feed_dict={self.TaskNextData:Data,self.TaskNextLabel:Label})                
                if EarlyStopFlag:
                    es.AddLoss(1-acc)
                if i%100==0:
                    self.Log("Task Net Loss %f Acc %f"%(loss,acc))
                    print(acc)
                    if es.ShouldStop():
                        break
        EvalEpoch=20
        acc=[]
        for i in range(EvalEpoch):
            Data,Label=self.TaskDataIter.NextBatch(Test=True)
            Acc=sess.run(self.TaskAcc,feed_dict={self.TaskNextData:Data,self.TaskNextLabel:Label})
            acc.append(Acc)
        acc=np.sum(acc)/EvalEpoch
        self.Log("Task Net Acc %f"%acc)
        return [loss],[acc]
   