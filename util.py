import numpy as np

def filter_pp_name(p):
	if p == 'pds' or p == 'pdc' or p == 'ppb' or p == 'ppr' or p == 'psd':
		p = 'pp'
	elif p == 'pfl':
		p = 'dem'
	elif p == 'mdb':
		p = 'pmdb'
	elif p == 'pr':
		p = 'pl'
	# elif p == 'prb' or p == 'pmr':
	# 	p = 'republicanos'
	return p

def get_valid_pp_single_net(net,top_n=None,cut_percent=None):
	pp = net.vs['political_party']
	pp = np.asarray(pp)
	unique,count = np.unique(pp,return_counts=True)
	idxs = np.argsort(-count)
	unique = unique[idxs]
	count = count[idxs]
	total = sum(count)
	valid_pp = []
	if cut_percent:
		cut = cut_percent*total
		for c,u in zip(count,unique):
			if c > cut:
				valid_pp.append(u)
		return valid_pp
	elif top_n:
		valid_pp = unique[:top_n]
		return valid_pp

def get_valid_pp(nets,begin,delta,top_n=3,cut_percent=None):
	top3 = set()
	filenames = []
	i = begin + delta
	for net in nets:
		valid_pp = get_valid_pp_single_net(net,top_n=top_n,cut_percent=cut_percent)
		print(i,valid_pp)
		i += delta
		top3 |= set(valid_pp)
	return top3