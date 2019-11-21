#!/usr/bin/python
# -*- coding: <utf-8> -*-
from igraph import Graph;
import io;
import numpy as np;
import re;
import sys

def __textSplit2(text):
	entries = text.split()
	if(len(entries)<2):
		raise ValueError();
	return (float(entries[0]),float(entries[1]));

def __textSplit3(text):
	entries = text.split()
	if(len(entries)<3):
		raise ValueError();
	return (float(entries[0]),float(entries[1]),float(entries[2]));

def __readNumberIgnoringNone(value):
	if(value.lower()=="none"):
		return 0;
	else:
		return float(value);


__propertyHeaderRegular = re.compile("#([ve]) \"(.+)\" ([sn]|v2|v3)");
__propertyFunctions = {
	"s": str,
	"n": __readNumberIgnoringNone,
	"v2": __textSplit2,
	"v3": __textSplit3
}

def __readXnetVerticesHeader(fp,currentLineIndex,lastLine=""):
	headerLine = lastLine;
	if(len(headerLine)==0):
		for lineData in fp:
			headerLine = lineData.decode("utf-8").rstrip();
			currentLineIndex+=1;
			if(len(headerLine)>0):
				break;

	headerEntries = headerLine.split();
	nodeCount = 0;

	if(len(headerEntries)==0 or headerEntries[0].lower() != "#vertices"):
		raise ValueError("Malformed xnet file [%s:%d]\n\t>%s"%(fp.name,currentLineIndex,headerLine));

	try:
		nodeCount = int(headerEntries[1]);
	except ValueError:
		raise ValueError("Malformed xnet file [%s:%d]\n\t>%s"%(fp.name,currentLineIndex,headerLine));

	return (nodeCount,currentLineIndex,"");

def __readXnetNames(fp,currentLineIndex,lastLine=""):
	names = [];
	fileEnded = True;
	for lineData in fp:
		lineString = lineData.decode("utf-8").rstrip();
		currentLineIndex+=1;
		
		if(len(lineString)==0):
			continue;
		
		lastLine = lineString;
		
		if(len(lineString)>0 and lineString[0]=="#"):
			fileEnded = False;
			break;
		
		if(len(lineString)>1 and lineString[0]=="\"" and lineString[-1]=="\""):
			lineString = lineString[1:-1];
		
		names.append(lineString);

	if(fileEnded):
		lastLine = "";
	return (names,currentLineIndex,lastLine);


def __readXnetEdgesHeader(fp,currentLineIndex,lastLine=""):
	headerLine = lastLine;
	if(len(headerLine)==0):
		for lineData in fp:
			headerLine = lineData.decode("utf-8").rstrip();
			currentLineIndex+=1;
			if(len(headerLine)>0):
				break;
	
	headerEntries = headerLine.split();
	
	weighted = False;
	directed = False;

	if(len(headerEntries)==0 or headerEntries[0].lower() != "#edges"):
		raise ValueError("Malformed xnet file [%s:%d]\n\t>%s"%(fp.name,currentLineIndex,headerLine));

	for headerEntry in headerEntries:
		if(headerEntry == "weighted"):
			weighted = True;
		if(headerEntry == "nonweighted"):
			weighted = False;
		if(headerEntry == "directed"):
			directed = True;
		if(headerEntry == "undirected"):
			directed = False;

	return ((weighted,directed),currentLineIndex,"");

def __readXnetEdges(fp,currentLineIndex,lastLine=""):
	edges = [];
	weights = [];
	fileEnded = True;
	for lineData in fp:
		lineString = lineData.decode("utf-8").rstrip();
		currentLineIndex+=1;
		
		if(len(lineString)==0):
			continue;
		
		lastLine = lineString;
		
		if(len(lineString)>0 and lineString[0]=="#"):
			fileEnded=False;
			break;
		
		entries = lineString.split();

		if(len(entries)<2):
			raise ValueError("Malformed xnet file [%s:%d]\n\t>%s"%(fp.name,currentLineIndex,headerLine));
		try:
			edge = (int(entries[0]),int(entries[1]));
			weight = 1.0;
			if(len(entries)>2):
				weight = float(entries[2]);
			edges.append(edge);
			weights.append(weight);
		except ValueError:
			raise ValueError("Malformed xnet file [%s:%d]\n\t>%s"%(fp.name,currentLineIndex,headerLine));

	if(fileEnded):
		lastLine = "";
	return ((edges,weights),currentLineIndex,lastLine);

def __readXnetPropertyHeader(fp,currentLineIndex,lastLine=""):
	global __propertyHeaderRegular;
	headerLine = lastLine;
	if(len(headerLine)==0):
		for lineData in fp:
			headerLine = lineData.decode("utf-8").rstrip();
			currentLineIndex+=1;
			if(len(headerLine)>0):
				break;

	headerEntries = __propertyHeaderRegular.findall(headerLine);

	if(len(headerEntries)==0 or (len(headerEntries)==1 and len(headerEntries[0])!=3)):
		raise ValueError("Malformed xnet file [%s:%d]\n\t>%s"%(fp.name,currentLineIndex,headerLine));
	
	(propertyType,propertyName,propertyFormat) = headerEntries[0];
	return ((propertyType,propertyName,propertyFormat),currentLineIndex,"");

def __readXnetProperty(fp,propertyFormat,currentLineIndex,lastLine=""):
	global __propertyFunctions;
	properties = [];
	propertyFunction = __propertyFunctions[propertyFormat];
	fileEnded = True;
	for lineData in fp:
		lineString = lineData.decode("utf-8").rstrip();
		currentLineIndex+=1;
		
		if(len(lineString)==0):
			continue;
		
		lastLine = lineString;
		
		if(len(lineString)>0 and lineString[0]=="#"):
			fileEnded = False;
			break;
		
		if(len(lineString)>1 and lineString[0]=="\"" and lineString[-1]=="\""):
			lineString = lineString[1:-1];

		try:
			properties.append(propertyFunction(lineString));
		except ValueError:
			raise ValueError("Malformed xnet file [%s:%d]\n\t>%s"%(fp.name,currentLineIndex,lineString));

	if(fileEnded):
		lastLine = "";

	return (properties,currentLineIndex,lastLine);

def xnet2igraph(fileName='test.xnet'):
	"""
	Read a Graph from a xnet formatted file.
	
	Parameters
	----------
	fileName : string
	    Input file path.
	"""
	network = None;
	with open(fileName, 'rb') as fp:
		currentLineIndex = 0;
		lastLine = "";
		(nodeCount,currentLineIndex,lastLine) = __readXnetVerticesHeader(fp,currentLineIndex,lastLine);
		(names,currentLineIndex,lastLine) = __readXnetNames(fp,currentLineIndex,lastLine);
		if(len(names)>0 and len(names)<nodeCount):
			raise ValueError("Malformed xnet file [%s:%d]\n\t>%s [%d entries expected but only %d found]"%(fp.name,currentLineIndex,headerLine,nodeCount,len(names)));
		((weighted,directed),currentLineIndex,lastLine) = __readXnetEdgesHeader(fp,currentLineIndex,lastLine);
		((edges,weights),currentLineIndex,lastLine) = __readXnetEdges(fp,currentLineIndex,lastLine);
		network = Graph(nodeCount,edges=edges,directed=directed);
		if(len(names)>0):
			network.vs["name"] = names;
		if(weighted):
			network.es["weight"] = weights;
		while(lastLine!=""):
			((propertyType,propertyName,propertyFormat),currentLineIndex,lastLine) = __readXnetPropertyHeader(fp,currentLineIndex,lastLine);
			(properties,currentLineIndex,lastLine) = __readXnetProperty(fp,propertyFormat,currentLineIndex,lastLine);
			if(propertyType=="e"):
				network.es[propertyName] = properties;
			elif(propertyType=="v"):
				network.vs[propertyName] = properties;
	return network;



def igraph2xnet(g,fileName='test.xnet',ignoredNodeAtts=[],ignoredEdgeAtts=[]):
	"""
	Write igraph object to .xnet format.
	
	Vertex attributes 'name' and 'weight' are treated in a special manner. They
	correspond to attributes assigned inside the #vertices tag.
	
	Edge attribute 'weight' is assigned to edges inside the #edges tag.
	
	Parameters
	----------
	g : igraph.Graph
	    Input graph.
	fileName : string
	    Output file.
	ignoredNodeAtts : list
	    List of node attributes to ignore when writing graph.
	ignoredEdgeAtts : list
	    List of edge attributes to ignore when writing graph.
	"""

	N = g.vcount()
	E = g.ecount()
	
	nodesAtt = g.vs.attributes()
	edgesAtt = g.es.attributes()

	if ('weight' in nodesAtt) and ('weight' not in ignoredNodeAtts):
		isNodeWeighted = True
		isNodeWeightedString = 'weighted'
	else:
		isNodeWeighted = False
		isNodeWeightedString = 'nonweighted'
		
	if ('name' in nodesAtt) and ('name' not in ignoredNodeAtts):
		isNodeNamed = True
	else:
		isNodeNamed = False		

	fd = open(fileName,'w')
	fd.write('#vertices '+str(N)+' '+isNodeWeightedString+'\n')

	if isNodeNamed and isNodeWeighted:
		for i in range(N):
			fd.write('\"'+g.vs[i]['name']+'\" '+str(g.vs[i]['weight'])+'\n')
	elif isNodeNamed and (not isNodeWeighted):
		for i in range(N):
			fd.write('\"'+g.vs[i]['name']+'\"'+'\n')
	elif (not isNodeNamed) and isNodeWeighted:
		for i in range(N):
			fd.write(str(g.vs[i]['weight'])+'\n')
			

	if ('weight' in edgesAtt) and ('weight' not in ignoredEdgeAtts):
		isEdgeWeighted = True
		isEdgeWeightedString = 'weighted'
	else:
		isEdgeWeighted = False		
		isEdgeWeightedString = 'nonweighted'

	if g.is_directed()==True:
		isEdgeDirected = True
		isEdgeDirectedString = 'directed'
	else:
		isEdgeDirected = False			
		isEdgeDirectedString = 'undirected'
	
	fd.write('#edges '+isEdgeWeightedString+' '+isEdgeDirectedString+'\n')
	
	for i in range(E):
			
		edge = g.es[i].tuple
		if isEdgeWeighted:
			fd.write(str(edge[0])+' '+str(edge[1])+' '+str(g.es[i]['weight'])+'\n')
		else:			
			fd.write(str(edge[0])+' '+str(edge[1])+'\n')
		
	if isNodeWeighted:	
		nodesAtt.remove('weight')
	if isNodeNamed:
		nodesAtt.remove('name')
	if isEdgeWeighted:
		edgesAtt.remove('weight')
		
	nodesAtt.sort()
	edgesAtt.sort()
	
	isPython2 = sys.version_info[0] == 2
	if isPython2:
		stringType = basestring
	else:
		stringType = str
				
	for att in nodesAtt:
		if att not in ignoredNodeAtts:
			sample = g.vs[0][att]
						
			typeSample = type(sample)
			if isinstance(sample, stringType):
				typeSampleString = 's'			
			elif np.isscalar(sample)==True:
				typeSampleString = 'n'
			elif (typeSample==list) or (typeSample==tuple) or (typeSample==np.ndarray):
				if len(sample)==2:
					typeSampleString = 'v2'
				elif len(sample)==3:
					typeSampleString = 'v3'
				
			fd.write('#v \"'+att+'\" '+typeSampleString+'\n')
			for i in range(N):
				if typeSampleString == 'n':
					fd.write(str(g.vs[i][att])+'\n')
				elif typeSampleString == 'v2':
					fd.write(str(g.vs[i][att][0])+' '+str(g.vs[i][att][1])+'\n')
				elif typeSampleString == 'v3':
					fd.write(str(g.vs[i][att][0])+' '+str(g.vs[i][att][1])+' '+str(g.vs[i][att][2])+'\n')
				elif typeSampleString == 's':
					fd.write('\"'+g.vs[i][att]+'\"'+'\n')
				

	for att in edgesAtt:
		if att not in ignoredEdgeAtts:
			sample = g.es[0][att]
			
			typeSample = type(sample)
			if isinstance(sample, stringType):
				typeSampleString = 's'			
			elif np.isscalar(sample)==True:
				typeSampleString = 'n'
			elif (typeSample==list) or (typeSample==tuple) or (typeSample==np.ndarray):
				if len(sample)==2:
					typeSampleString = 'v2'
				elif len(sample)==3:
					typeSampleString = 'v3'
				
			fd.write('#e \"'+att+'\" '+typeSampleString+'\n')
			for i in range(E):
				if typeSampleString == 'n':
					fd.write(str(g.es[i][att])+'\n')
				elif typeSampleString == 'v2':
					fd.write(str(g.es[i][att][0])+' '+str(g.es[i][att][1])+'\n')
				elif typeSampleString == 'v3':
					fd.write(str(g.es[i][att][0])+' '+str(g.es[i][att][1])+' '+str(g.es[i][att][2])+'\n')
				elif typeSampleString == 's':
					fd.write('\"'+g.es[i][att]+'\"'+'\n')				
				
	fd.close()	
	
def igraph2xnet_fast(g):
	"""
	Convert igraph object to a .xnet format string. This string 
	can then be written to a file or used in a pipe. The purpose 
	of this function is to have a fast means to convert a graph
	without any attributes to the .xnet format.
	
	Parameters
	----------
	g : igraph.Graph
	    Input graph.
	
	Returns
	-------
	strOut : string
	    String that can be written to a .xnet file.
	"""

	N = g.vcount()
	strOut = "#vertices "+str(N)+" nonweighted\n#edges nonweighted undirected"
	
	edList = np.array(g.get_edgelist())
	
	s = io.BytesIO()
	np.savetxt(s, edList, fmt='%d', header=strOut, comments='')
	
	strOut = s.getvalue()
	s.close()
	
	return strOut	
	
	
# Simple test if running as script	
if __name__ == '__main__':	
	g = Graph(edges=[[0,1],[0,2],[1,2],[2,3],[3,4],[1,5],[1,6]],directed=False)

	g.vs['name'] = ['Node 0','Node 1','Node 2','Node 3','Node 4','Node 5','Node 6']	
	g.vs['weight'] = [1.,4.,2.,0.,1.,5.,3.] 

	g.vs['community'] = [0,0,0,1,1,2,3]
	g.vs['pos'] = [[1.2,3.],[0.2,0.1],[0.4,0.6],[2.3,5.5],[2.4,0.4],[10.2,3.],[5.,5.]]
	g.vs['type'] = ['Crazy','Crazy','Normal','Sausage','Cat','Batman','Mouse']

	g.es['weight'] = [0.,-4.,5.,3.,2.,10.,3.]
	g.es['name'] = ['Edge 0','Edge 1','Edge 2','Edge 3','Edge 4','Edge 5','Edge 6']	
	g.es['color'] = [[0.1,0.1,0.1],[0.,1.,0.],[1.,0.,1.],[1.,1.,0.],[0.2,0.4,0.6],[0.1,0.5,0.9],[0.5,0.5,0.5]]

	igraph2xnet(g,'test.xnet',[],[])
	
	stringGraph = igraph2xnet_fast(g)


	