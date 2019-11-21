import glob
import xnet
import igraph
import leidenalg # pip install leidenalg
import subprocess

def get_largest_component(g):
		components = g.components()
		giant = components.giant()
		return giant

def identify_communities_leidenalg(net):
	giant = get_largest_component(net)
	comms = leidenalg.find_partition(giant, leidenalg.ModularityVertexPartition)
	comm_list = comms.subgraphs() # communities in current level
	print('Number of communities identified:',len(comm_list))
	net_copy = net.copy()
	net_copy.vs['community'] = "-1"
	for idx,comm in enumerate(comm_list):
		for v1 in comm.vs:
			v2 = net_copy.vs.find(name=v1['name'])
			v2['community'] = str(idx+1)
	return net_copy		

filenames = glob.glob("data/1991-2019/by_year/*.xnet")
filenames = sorted(filenames)

graphs = []	
for filename in filenames:
	print(filename)
	net = xnet.xnet2igraph(filename)
	net = identify_communities_leidenalg(net)

	output = filename[:-5] + '_leidenalg.xnet'
	xnet.igraph2xnet(net,output)