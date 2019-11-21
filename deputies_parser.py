import os
import ast
import xnet
import json
import pickle
import zipfile
import concurrent.futures

import numpy as np 
import cairocffi as cairo
import matplotlib.pyplot as plt

from igraph import Graph
from collections import defaultdict, namedtuple

Deputy = namedtuple('Deputy', ['name', 'political_party'])

def get_votes(common_votes, prop):

    votes = prop['votos']

    # para evitar votos repetidos
    votes = [str(v) for v in votes]
    votes = list(set(votes))
    votes = [dict(ast.literal_eval(v)) for v in votes]
    
    yes = [] # deputados que votaram sim
    no = [] # deputados que votaram não
    abst = []
    obst = []
    for v in votes:
        d = Deputy(v['nome'],v['partido'])
        if v['voto'] == 'Sim':
            yes.append(d)
        elif v['voto'] == 'Não':
            no.append(d)
        elif v['voto'] == 'Abstenção':
            abst.append(d)
        elif v['voto'] == 'Obstrução':
            obst.append(d)
    all_votes = yes + no + abst + obst
    for d0 in all_votes:
        for d1 in all_votes:
            if d0 == d1:
                break
            if (d0 in yes and d1 in yes) or (d0 in no and d1 in no) or (d0 in abst and d1 in abst) or (d0 in obst and d1 in obst):
                common_votes[frozenset([d0,d1])] += 1
            else:
                common_votes[frozenset([d0,d1])] -= 1

def set_info(v0, names, political_parties):
	if not v0.name in names:
		names.append(v0.name)
		political_parties.append(v0.political_party)

def generate_graph(common_votes, w_total):
	names = []
	political_parties = []
	edges = []
	weights = []
	for e,w in common_votes.items():
		v0,v1 = tuple(e)
		set_info(v0, names, political_parties)
		set_info(v1, names, political_parties)	
		if w > 0: # só cria a aresta se a quantidade de votos em comum for positiva
			v0_id = names.index(v0.name)
			v1_id = names.index(v1.name)
			edges.append((v0_id,v1_id))
			weights.append(w/w_total)

	g = Graph()
	g.add_vertices(len(names))
	g.add_edges(edges)
	g.es['weight'] = weights
	g.vs['name'] = names
	g.vs['political_party'] = political_parties
	return g

def generate_graph_year_3months(search,year,month,transition_years):
	common_votes = defaultdict(lambda:0)
	n_props = 0
	props = search[year] + search[year + 1]
	for prop in props:
		_,mm,yyyy = tuple(prop['data'].split('/'))
		mm = int(mm)
		yyyy = int(yyyy)

		if yyyy == year and (mm == month or mm == month + 1 or mm == month + 2):
			# print(mm,yyyy,end=', ',sep='/')
			get_votes(common_votes,prop)
			n_props +=1
		elif not year in transition_years:
			if month + 2 > 12 and mm == (month + 2)%12 and yyyy == (year + 1):
				# print(mm,yyyy,end=', ',sep='/')
				get_votes(common_votes,prop)
				n_props +=1
			elif month + 1 > 12 and mm == (month + 1)%12 and yyyy == (year + 1):
				# print(mm,yyyy,end=', ',sep='/')
				get_votes(common_votes,prop)
				n_props +=1
		
	if n_props > 6:
		g = generate_graph(common_votes, n_props)
		return g
	else:
		print('Problem in year',year,'and month',month)
		print('Number of propositions',n_props)
		return None


def get_nets_by_year(search,years,output_dir):
    for year in years:
        common_votes = defaultdict(lambda:0)
        props = search[year]
        n_props = len(props)
        for prop in props:
            get_votes(common_votes,prop)
        if n_props > 6:
            g = generate_graph(common_votes, n_props)
            xnet.igraph2xnet(g, output_dir+'dep_'+str(year)+'_obstr.xnet')
        else:
            print('Problem in year',year)
            print('Number of propositions',n_props)
        
def get_nets_by_3months(search,year,output_dir):
    for month in range(1,13): # 12 meses
        print("Current year:", year,"Current month:",month)
        g = generate_graph_year_3months(search,year,month,transition_years)
        if g:
            xnet.igraph2xnet(g, output_dir+'dep_'+str(year)+'_'+str(month)+'.xnet')
        print()

if __name__ == '__main__':
    
    with zipfile.ZipFile('data/deputadosv2.zip', 'r') as zip_ref:
        zip_ref.extractall('data/')

    input_file = "data/deputadosv2.json"
    file = open(input_file, 'r').read()
    propositions = json.loads(file)['proposicoes']
    print("Total propositions:", len(propositions))

    search = defaultdict(lambda:[])
    for prop in propositions:
        year = prop['data'].split('/')[2]
        year = int(year)
        search[year].append(prop)

    output_dir = 'data/1991-2019/'

    # Geração de grafos por ano:

    years = list(range(1991,2020))
    get_nets_by_year(search,years,output_dir+'by_year/')

    # Geração dos grafos por mandato (com janela de 3 em 3 meses):

    years = list(range(1997,2020)) # primeiro mandato considerado
    transition_years = list(range(1998,2020,4)) # ano de eleição (véspera de início de novo mandato)

    for year in years:
        get_nets_by_3months(search,1996,output_dir+'mandate/')
