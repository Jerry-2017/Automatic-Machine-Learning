def SortTupleList(TupleList,StartFrom=0):
	
	def LessThan(Tuple1,Tuple2):
		LengthofTuple1=len(Tuple1)
		LengthofTuple2=len(Tuple2)
		CommonLength=min(LengthofTuple1,LengthofTuple2)
		for i in range(0,CommonLength):
			if Tuple1[i]<Tuple2[i]:
				return True
		if LengthofTuple1<LengthofTuple2:
			return True
		else:
			return False
	
	LengthofList=len(TupleList)
	
	for i in range(LengthofList):
		for j in range(i+1,LengthofList):
			if LessThan(TupleList[j],TupleList[i]):
				TupleExchange=TupleList[i]
				TupleList[i]=TupleList[j]
				TupleList[j]=TupleExchange
				
	return TupleList
				
				
			