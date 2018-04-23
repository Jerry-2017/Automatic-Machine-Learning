from itertools import combinations, permutations
from Common import SortTupleList

class Operator:
	def __init__(self,InputList,InputDegree,OutputDegree,Name):
		self.InputDegree=InputDegree
		self.OutputDegree=OutputDegree
		self.Name=Name
		self.Tensor=self.ConstructFunc(InputList)
		
	def ConstructFunc(self,InputList):
		assert False,'Not Implemented'
	
	def GetInputDegree(self):
		return self.InputDegree
		
	def GetOutputDegree(self):
		return self.OutputDegree
		
	def GetName(self):
		return self.Name
	
	def GetTensor(self):
		return self.Tensor
	
	def GetParameters(self):
		return None
	
	def SetParameters(self):
		assert False,'Not Implemented'
	
class Input(Operator):
	def __init__(self,_Input):
		Super(Input,self).__init__(InputList=[_Input],InputDegree=0,OutputDegree=1,Name='Input')
		
	def ConstructFunc(self,InputList):
		self.Tensor=InputList[0]

class NoConnection(Operator):
	def __init__(self,_Input=None):
		Super(NoConnection,self).__init__(InputList=[_Input],InputDegree=0,OutputDegree=0,Name='NULL')
		
	def ConstructFunc(self,InputList):
		assert False,'Should not appear.'

class Output(Operator):
	def __init__(self,_Input):
		Super(Output,self).__init__(InputList=_Input,InputDegree=1,OutputDegree=0,Name='Output')
		
	def ConstructFunc(self,InputList):
		self.Tensor=InputList[0]
		
class Graph:
	def __init__(self,VertexNum,OperatorList,InputNum=1,OutputNum=1,ConcatOperator=None):
		assert ConcatOperator is not None
		
		self.VertexNum=VertexNum
		self.OperatorList=[NoConnection]+OperatorList
		self.InputNum=InputNum
		self.OutputNum=OutputNum
		self.VertexList=[None]*self.VertexNum
		self.VertexOpType=[None]*self.VertexNum
		self.IncidenceMatrix=[[None for i in range(self.VertexNum)] for j in range(self.VertexNum)]
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
		
	
	def ConnectOptions(self):
	
		OuputDegreeSum=0
		InputFromList=[] 
		# The all possible inputs a new operator could takes
		
		for i in range(self.VertexOccupied):
			if self.VertexDegreeQuota[i]>0:
				OutputDegreeSum+=1
				InputFromList.append(i)
		ConnectionOptions=[]
		#The all possible connections a new opeartor could takes
		# [OperatorTypeNum, NumberOfInputs, NumberOfOutputs Input1, Input2, ...]
		
		for OperatorIndex in self.OperatorList:
			OperatorInputDegree=self.OperatorList[OperatorIndex].GetInputDegree()
			OperatorOutputDegree=self.OperatorList[OperatorIndex].GetOutputDegree()
			if TempOperatorDegree<=OutputDegreeSum():
			
				for InputComb in list( combinations(InputFromList), OperatorInputDegree): 
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
		
		self.OperatorApplied=OperatorApplied[:-1]
	
	def UnifiedTransform(self,GraphType='2D'):
		"""The output of unified graph could include 2D Convolution & 3D Convolution"""
		TupleList=[]
		for i in range(self.VertexVertexOccupied):
			Tuple=[i,self.VertexDepth[i],self.VertexInputDegree[i]]
			for InputVertex in self.VertexInputList[i]:
				Tuple.append(self.VertexInputDegree[InputVertex])
			TupleList.append(Tuple)
		TupleList=SortTupleList(TupleList)
		NewGraph=[ [None for i in range(self.VertexNum)] for i in range(VertexNum)]
		if GraphType=='2D':
			for i in TupleList:
				for j in TupleList:
					NewGraph[0][0]=self.IncidenceMatrix[i[0]][j[0]]
		elif GraphType='3D':
			assert False,error
		return NewGraph
		
	
	def BuildGraph(self,Inputs):
		"""TensorBuild?"""
		InternalTensor=[None for i in range(self.VertexNum)]
		InternalIndex=len(Inputs)
		for i in range(len(Inputs)):
			InternalTensor[i]=Input(Inputs[i])
		
		for Option in self.OperatorApplied:
			TensorInput=[]
			for Index in Option[3:]:
				TensorInput.append(InternalTensor[Index])
			InternalTensor[InternalIndex]=self.OperatorList[Option[0]]().ConstructFunc(*TensorInput)
			
		ToBeConnected=[]
		for i in range(self.VertexOccupied):
			if self.VertexDegreeQuota[i]>0:
				ToBeConnected+=[self.OperatorList[i] for _ in range(self.VertexDegreeQuota[i])]:
				
		Output=self.ConcatOperator().ConstructFunc(ToBeConnected)
		return Output
		
	def GetParameterCopy(self):
		ParameterList=[]
		for i in range(
				