import tensorflow as tf
from Operator import *
from ImageOperators import *
from Graph import Graph
Data_Format='NHWC'

class QLearning():
    def __init__(self):
        self.sess=tf.session()
        pass
        
    def ConstructQFunc3D(self,ImageSize=5,BitDepth=7,BatchSize=5):
        
        self.QNetData = tf.placeholder(tf.float32 , shape=[None,7,5,5,1],name="input")
        self.QNetLabel = tf.placeholder(tf.float32, shape=[None], name="label" )
        
        DatasetRaw = tf.data.Dataset.from_tensor_slices((Data,Label))
        Dataset=DatasetRaw.repeat().batch(BatchSize)
        DataIter = Dataset.make_initializable_iterator()        
        NextData,NextLabel = DataIter.get_next()    
    
        self.IncidenceMatrix=tf.Variable("Input")
        self.Label=tf.Variable("Label")
        
        Conv3D_Layer1_Kernel=tf.placeholder([BitDepth,2,2,1,3])
        #[filter_depth, filter_height, filter_width, in_channels, out_channels]
        Conv3D_Layer1=tf.nn.conv3d( input=self.IndcidenceMatrix,
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
        
        Output_Layer=tf.layers.dense(inputs=Reshape_Layer3, units=1, activation=tf.nn.sigmoid)

        self.Output=Output_Layer
        
        self.Loss = tf.losses.mean_squared_error(labels=self.QNetLabel, predictionss=self.QNetOutput)
        
        self.Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        self.Train_op = self.Optimizer.minimize(
                loss=self.Loss,
                global_step=tf.train.get_global_step())           
    
    def SetNetworkGenerator(self,NetworkGenerator):
        self.NetworkGenerator=NetworkGenerator
    
    def SetOperatorList(self,OperatorList):
        self.OperatorList=OperatorList
    
    def BuildTaskGraph(self,Graph,Data,Label,BatchSize,OutputDecor):
    
        DatasetRaw = tf.data.Dataset.from_tensor_slices((Data,Label))
        Dataset=DatasetRaw.repeat().batch(BatchSize)
        DataIter = Dataset.one_shot_iterator()      
        NextData,NextLabel = DataIter.get_next()      
        Temp_Output=Graph.BuildGraph(NextData)
        self.TaskOutput,self.TaskLoss=OutputDecor(Output,NextLabel)
        self.TaskTrain = self.Optimizer.minimize(
                loss=self.TaskLoss,
                global_step=tf.train.get_global_step())        ## Need to Change
        
    
    def StartTrial(self,TaskSpecific):
        LogHistory=TaskSpecific["LogHistory"]
        OperatorList=TaskSpecific["OperatorList"]
        NetworkDecor=TaskSpecific["NetworkDecor"]
        VertexNum=TaskSpecific["OperatorNum"]
        InputNum=TaskSpecific["InputNum"]
        OutputNum=TaskSpecific["OutputNum"]
        BatchSize=TaskSpecific["BatchSize"]

        TaskInput=TaskSpecific["TaskInput"]
        TaskLabel=TaskSpecific["TaskLabel"]
        self.HisNet=[]
        self.HisNetPerf=[]
        for i in range(Times):
            Gph=Graph(  VertexNum=VertexNum,
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
            
            BuiltNet=self.BuildGraph(BatchSize)
            self.TrainTaskNet(Step=100)
            
            Step=100
            self.BuildTaskGraph(Graph=Gph,Data=TaskInput,Label=TaskLabel,OutputDecor=NetworkDecor)
            self.TrainTaskGraph()
            
            HisItem={"OptionList":Gph.GetOptionList(),"TrainStep":Step,"Performance":Performance,"UnifiedNet":Gph.UnifiedTransform('3D')}
            if LogHistory==True:
                self.Log(HisItem)
            self.HisNet.append(HisItem["UnifiedNet"])
            self.HisNetPerf.append(HisItem["Performance"])
            QStep=100
            self.TrainQNet(self.Output,self.HisNet,self.HisNetPerf,QStep)
    
    def Log(self,Log):
        print(Log)
                
    def TrainQNet(self,Output,Data,Label,Step):
        sess=self.sess
        sess.run(tf.global_variables_initializer())
        for i in range(Step):                
            _,acc=sess.run([self.Train_op,self.Loss],feed_dict={"input":QNetData,"label":QNetLabel})
            print(acc)
                
    def TrainTaskNet(self,Step=100):
        sess=self.sess
        sess.run(tf.global_variables_initializer())
        for i in range(Step):                
            _,acc=sess.run([self.TaskTrain,self.TaskLoss])
            print(acc)
   