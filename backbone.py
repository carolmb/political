import glob
import xnet
import numpy as np

from igraph import *
from mpmath import mp # pip install mpmath
from scipy import integrate

mp.dps = 50

# source: https://github.com/aekpalakorn/python-backbone-network/blob/master/backbone.py
def disparity_filter(g):
	total_vtx = g.vcount()
	g.es['alpha_ij'] = 1

	for v in range(total_vtx):
		edges = g.incident(v)

		k = len(edges)
		if k > 1:
			sum_w = mp.mpf(sum([g.es[e]['weight'] for e in edges]))
			for e in edges:
				w = g.es[e]['weight']
				p_ij = mp.mpf(w)/sum_w
				alpha_ij = 1 - (k-1) * integrate.quad(lambda x: (1-x)**(k-2), 0, p_ij)[0]
				g.es[e]['alpha_ij'] = min(alpha_ij,g.es[e]['alpha_ij'])

def alpha_cut(alpha,g):
	g_copy = g.copy()
	to_delete = g_copy.es.select(alpha_ij_ge=alpha)
	g_copy.delete_edges(to_delete)
	return g_copy

def get_largest_component_size(g):
	components = g.components()
	giant = components.giant()
	return giant.vcount()

def get_best_cut(net,preserve_percent,a_min,a_max):
		a_min = mp.mpf(a_min)
		a_max = mp.mpf(a_max)

		error = 0.015
		largest_size = get_largest_component_size(net)

		min_erro = 1000
		a_min_erro = 0.0

		def get_current_percent(a):
			nonlocal min_erro, a_min_erro, a_min, a_max
			cuted_net = alpha_cut(a,net)
		# print('number of edges',cuted_net.ecount())

			preserved_size = get_largest_component_size(cuted_net)
		# print('preserved size',preserved_size)

			current_percent = mp.mpf(preserved_size)/mp.mpf(largest_size)

			if min_erro > abs(current_percent-preserve_percent):
				min_erro = abs(current_percent-preserve_percent)
				a_min_erro = a

			return cuted_net,current_percent,a

		i = 0

		a_min_perc = mp.mpf(get_largest_component_size(alpha_cut(a_min,net)))/mp.mpf(largest_size)
		a_max_perc = mp.mpf(get_largest_component_size(alpha_cut(a_max,net)))/mp.mpf(largest_size)

		a = 0.0

		while True:
			if i > 100:
				cuted_net = alpha_cut(a_min_erro,net)

				print('error infinity loop')
				print('alpha %.2f; preserved %.2f' % (a_min_erro,min_erro+preserve_percent))
				print()
				return cuted_net

			i += 1
				
			a = (a_min+a_max)/2

			cuted_net,current_percent,a = get_current_percent(a)
			current_erro = current_percent-preserve_percent
			
			if abs(current_erro) < error:
				print('total iterations to find the graph',i)
				print('alpha %.2f; preserved %.2f' % (a,current_percent))
				print()

				return cuted_net

			if (a_min_perc-preserve_percent)*(current_percent-preserve_percent) > 0:
				a_min = a
				a_min_perc = current_percent
			else:
				a_max = a
				a_max_perc = current_percent

def apply_backbone(net,a_min,a_max,preserve=0.8):
	disparity_filter(net)
	best = get_best_cut(net,preserve,a_min,a_max)
	return best

if __name__ == '__main__':
	filenames = glob.glob('data/1991-2019/mandate/*.xnet')
	filenames = sorted(filenames)
	print(filenames)

	preserve = 0.8
	a_min = 0.0001
	a_max = 1
	for filename in filenames:
		print(filename)
		net = xnet.xnet2igraph(filename)

		if net.ecount() > 0: # 2002_6 problem
			net = apply_backbone(net,a_min,a_max,preserve)
			output = filename[:-5] + '_' + str(preserve) + '.xnet'
			xnet.igraph2xnet(net,output)
