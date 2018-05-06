from itertools import combinations, permutations
from Common import SortTupleList
import numpy as np
import tensorflow as tf

class Operator:
    Name="GenericOp"
    InputDegree=1
    OutputDegree=1
    def __init__(self,InputList=None,ID="Default",BuildFlag=True):
        if InputList is not None and BuildFlag:
            self.ConstructFunc(InputList)
        else:
            self.Tensor=None
        self.ID="Name"
        
    def ConstructFunc(self,InputList):
        assert False,'Not Implemented'
    
    @classmethod
    def GetInputDegree(cls):
        return cls.InputDegree
    
    @classmethod
    def GetOutputDegree(cls):
        return cls.OutputDegree
    
    @classmethod
    def GetName(cls):
        return cls.Name
    
    def GetTensor(self):
        assert self.Tensor is not None
        return self.Tensor
    
    def GetParameters(self):
        return None
    
    def SetParameters(self,Parameter=None):
        if Parameter is not None:
            assert False,'Not Implemented'
            
    def CheckInputList(self,InputList=None):
        pass

class BackTrackError(Exception):
    pass
            
class Input(Operator):
    InputDegree=0
    OutputDegree=1
    Name='Input'
    def __init__(self,InputList,ID,BuildFlag=True):        
        super(Input,self).__init__(InputList=InputList,ID=ID,BuildFlag=BuildFlag)
        
    def ConstructFunc(self,InputList):
        self.Tensor=InputList[0]

class NoConnection(Operator):
    InputDegree=0
    OutputDegree=0
    Name='NULL'
    def __init__(self,InputList=None,ID=None,BuildFlag=True):
        super(NoConnection,self).__init__(InputList=InputList,ID=ID,BuildFlag=BuildFlag)
        
    def ConstructFunc(self,InputList):
        assert False,'Should not appear.'

class Output(Operator):
    InputDegree=1
    OutputDegree=0
    Name='Output'
    def __init__(self,_Input,ID=None,BuildFlag=True):
        super(Output,self).__init__(InputList=_Input,ID=ID,BuildFlag=BuildFlag)
        
    def ConstructFunc(self,InputList):
        self.Tensor=InputList[0]
        
class Graph:
    def __init__(self,VertexNum,OperatorList,InputNum=1,OutputNum=1,ConcatOperator=None,InputOperator=None):
        assert ConcatOperator is not None
        
        self.VertexNum=VertexNum
        self.OperatorList=[NoConnection]+OperatorList
        self.LenOpList=len(self.OperatorList)
        self.InputNum=InputNum
        self.OutputNum=OutputNum
        self.VertexList=[None]*self.VertexNum
        self.VertexOpType=[None]*self.VertexNum
        self.IncidenceMatrix=[[0 for i in range(self.VertexNum)] for j in range(self.VertexNum)]
        self.ConcatOperator=ConcatOperator
        
        self.OperatorApplied=[]
        self.VertexDepth=[0 for i in range(self.VertexNum)]
        self.VertexInputDegree=[0 for i in range(self.VertexNum)]
        self.VertexInputList=[[] for i in range(self.VertexNum)]
        self.VertexOutputDegree=[0 for i in range(self.VertexNum)]
        self.VertexOutputList=[[] for i in range(self.VertexNum)]
        self.OutputOccupied=0
        self.VertexOccupied=self.InputNum
        self.VertexDegreeQuota=[0 for i in range(self.VertexNum)]
        
        for i in range(self.InputNum):
            self.VertexDegreeQuota[i]=1
        
        if InputOperator is None:
            self.InputOperator=Input
        else:
            self.InputOperator=InputOperator
    
    def ConnectOptions(self):
    
        OutputDegreeSum=0
        InputFromList=[] 
        # The all possible inputs a new operator could takes
        
        for i in range(self.VertexOccupied):
            if self.VertexDegreeQuota[i]>0:
                OutputDegreeSum+=1
                InputFromList.append(i)
        ConnectionOptions=[]
        #The all possible connections a new opeartor could takes
        # [OperatorTypeNum, NumberOfInputs, NumberOfOutputs , Input1, Input2, ...]
        
        for OperatorIndex in range(1,len(self.OperatorList)):
            #print(self.OperatorList[OperatorIndex])
            OperatorInputDegree=self.OperatorList[OperatorIndex].GetInputDegree()
            OperatorOutputDegree=self.OperatorList[OperatorIndex].GetOutputDegree()
            if OperatorInputDegree<=OutputDegreeSum:
            
                for InputComb in list( combinations(InputFromList, OperatorInputDegree) ): 
                # combinations discard the differences between inputs, permutations verse vice 
                
                    Option=[OperatorIndex,OperatorInputDegree,OperatorOutputDegree]
                    Option.extend(InputComb)
                    ConnectionOptions.append(Option)
                
        return ConnectionOptions

    def ApplyOption(self,Option):
        VertexAdd=self.VertexOccupied
        self.VertexOccupied+=1
        assert( self.VertexOccupied < self.VertexNum )
        LastDepth=self.VertexNum
        self.VertexInputDegree[VertexAdd]=Option[1]
        
        for VertexFrom in Option[3:]:
            self.IncidenceMatrix[VertexFrom][VertexAdd]=Option[0]
            # The operator name from index 1
            LastDepth=min(LastDepth,self.VertexDepth[VertexFrom])
            self.VertexOutputDegree[VertexFrom]+=1
            self.VertexOutputList[VertexFrom].append(VertexAdd)
            self.VertexInputList[VertexAdd].append(VertexFrom)
            
        self.VertexDegreeQuota[VertexAdd]=Option[2]
        self.VertexOpType[VertexAdd]=Option[0]
        # Operator OutputDegree
        
        self.VertexDepth[VertexAdd]=LastDepth+1
        self.OperatorApplied.append(Option)

    def InitializeCheckOptionInput(self,Inputs):
        self.BuildGraph(Inputs)
        
    def CheckOption(self,Option):
        InputList=[]
        OperatorIndex=Option[0]
        for Input in Option[3:]:
            InputList.append(self.InternalTensor[Input])
        TempOperator=self.OperatorList[OperatorIndex](InputList,ID="",BuildFlag=False)
        return TempOperator.CheckValid(InputList)
            
    
    def RevokeOption(self,Option):
        self.VertexOccupied-=1
        VertexAdd=self.VertexOccupied
        assert( self.VertexOccupied >= self.InputNum)
        self.VertexInputList[VertexAdd]=[]
        self.VertexInputDegree[VertexAdd]=0
        for VertexFrom in Option[3:]:
            self.IncidenceMatrix[VertexFrom][VertexAdd]=0
            self.VertexOutputDegree[VertexFrom]-=1
            self.VertexOutputList[VertexFrom]=self.VertexOutputList[VertexFrom][:-1]
            
        self.VertexDegreeQuota[VertexAdd]=0
        
        self.OperatorApplied=self.OperatorApplied[:-1]
    
    def UnifiedTransform(self,GraphType='2D'):
        assert GraphType in ['3D','2D','3D_NoNull']
        """The output of unified graph could include 2D Convolution & 3D Convolution"""
        TupleList=[]
        for i in range(self.VertexOccupied):
            Tuple=[i,self.VertexDepth[i],self.VertexInputDegree[i]]
            for InputVertex in self.VertexInputList[i]:
                Tuple.append(self.VertexInputDegree[InputVertex])
            TupleList.append(Tuple)
        TupleList=SortTupleList(TupleList)      
        if GraphType=='2D':
            NewGraph=[ [None for i in range(self.VertexNum)] for i in range(VertexNum)]
            for i in TupleList:
                for j in TupleList:
                    NewGraph[0][0]=self.IncidenceMatrix[i[0]][j[0]]
        elif GraphType=='3D':
            NewGraph=np.zeros(shape=[self.LenOpList,self.VertexNum,self.VertexNum,1])
            for i in TupleList:
                for j in TupleList:
                    # Operation Number, Vertex Size, Vertex Size, Channel Size
                    NewGraph[self.IncidenceMatrix[i[0]][j[0]]][0][0][0]=1
        elif GraphType=='3D_NoNull':
            NewGraph=np.zeros(shape=[self.LenOpList-1,self.VertexNum,self.VertexNum,1])
            for i in TupleList:
                for j in TupleList:
                    if self.IncidenceMatrix[i[0]][j[0]]!=0:
                        NewGraph[self.IncidenceMatrix[i[0]][j[0]]-1][0][0][0]=1                    
                        
        return NewGraph
        
    
    def BuildGraph(self,Inputs):
        """TensorBuild?"""
        InternalTensor=[None for i in range(self.VertexNum)]
        InternalIndex=len(Inputs)
        for i in range(InternalIndex):
            InternalTensor[i]=self.InputOperator([Inputs[i]],ID="TaskNet_Input%d"%i)
        #print(InternalTensor[0].GetTensor().shape)
        for Option in self.OperatorApplied:
            TensorInput=[]
            for Index in Option[3:]:
                TensorInput.append(InternalTensor[Index])
            ScopeName="TaskNet_Node%d"%Index
            with tf.variable_scope(ScopeName,reuse=tf.AUTO_REUSE) as scope:              
                InternalTensor[InternalIndex]=self.OperatorList[Option[0]](InputList=TensorInput,ID=ScopeName)
            
        ToBeConnected=[]
        for i in range(self.VertexOccupied):
            if self.VertexDegreeQuota[i]>0:
                ToBeConnected+=[InternalTensor[i] for _ in range(self.VertexDegreeQuota[i])]
        print(ToBeConnected)
        Output=self.ConcatOperator(ToBeConnected)
        self.InternalTensor=InternalTensor
        return Output
        
    def GetParameter(self):
        ParameterList=[]
        for i in range(self.VertexOccupied):
            ParameterList.append(self.InternalTensor[i].GetParameter())
    
    def SetParameter(self,ParameterList):
        LenParamList=len(Parameter)
        assert(LenParamList>=self.VertexOccupied)
        for i in range(self.VertexOccupied):
            self.InternalTensor.SetParameter(ParamerterList[i])
        
             