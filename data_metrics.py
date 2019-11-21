import xnet
import glob

import numpy as np
import matplotlib.pyplot as plt

from igraph import *
from util import get_valid_pp
from util import filter_pp_name
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score

np.set_printoptions(suppress=True,formatter={'float_kind':'{:f}'.format})

header = 'imgs/'

def read_nets_by_years(path):
    filenames = glob.glob(path)
    filenames = sorted(filenames)

    dates = []
    nets = []
    for filename in filenames:
        net = xnet.xnet2igraph(filename)
        net.vs['political_party'] = [filter_pp_name(p) for p in net.vs['political_party']]
        nets.append(net.components().giant())

        base = filename.split('dep')[1].split('_')
#         date = float(filename.split('dep')[1].split('_')[0])
        date = float(filename.split('_')[2].split('.')[0])
        dates.append(date)

    dates = np.asarray(dates)
    nets = np.asarray(nets)

    sorted_idxs = np.argsort(dates)
    dates = dates[sorted_idxs]
    nets = nets[sorted_idxs]
    return dates,nets

def plot(xs,ys,x_name,y_name,filename,dim=(12,2)):

    plt.figure(figsize=dim)
    plt.plot(xs,ys,'o',ls='-')

    labels = [str(x) if float(x)%5 == 0 else '' for x in xs]
    plt.xticks(np.arange(min(xs), max(xs)+1, 1.0),labels=labels)  
    
    # plt.legend(loc='upper right')
    plt.xlabel(x_name)
    plt.ylabel(y_name)                
    plt.savefig(header+filename+'.pdf',format='pdf',bbox_inches="tight")
    plt.close()

def degree(dates,nets):
    degrees = [mean(net.degree()) for net in nets]
    plot(dates,degrees,'year','mean degree','mean_degree')

def dep_by_pp(dates,nets):
    ys = defaultdict(lambda:[])
    for date,net in zip(dates,nets):
        unique,count = np.unique(net.vs['political_party'],return_counts=True)
        for u,c in zip(unique,count):
            ys[u].append((date,c))
    total = [(sum([v for d,v in d_v]),k) for k,d_v in ys.items()]
    total = sorted(total,reverse=True)
    
    ys_sorted = []
    labels_sorted = []
    others = np.zeros(len(dates))
    for t,k in total:
        current_ys = ys[k]
        current_dates = [d for d,c in current_ys]
        real_y = []
        for date in dates:
            if not date in current_dates:
                real_y.append(0)
            else:
                real_y.append(current_ys[0][1])
                current_ys = current_ys[1:]
        if t <= 288:
            others += real_y
            continue
        labels_sorted.append(k)
        ys_sorted.append(real_y)
    
    ys_sorted = [others] + ys_sorted
    labels_sorted = ['others'] + labels_sorted
    
    ys_sorted = np.asarray(ys_sorted)
    ys_sorted = np.cumsum(ys_sorted, axis=0) 
    
    fig = plt.figure(figsize=(12,3))
    ax1 = fig.add_subplot(111)
    
    for label,pp_ys in (zip(reversed(labels_sorted),reversed(ys_sorted))):
        ax1.fill_between(dates, pp_ys, label=label.upper(),alpha=1)
    
    plt.legend(loc='upper right',bbox_to_anchor=(1.1, 1.0))
    plt.xlabel('year')
    plt.ylabel('cumulative number of deputies')
    plt.savefig(header+'cumulative_number_of_dep.pdf',format='pdf',bbox_inches="tight")
    plt.close()

def modularity(dates,nets,param1,param2):
    mods_pps = []
    mods_comm = []
    for net in nets:        
        pps = list(set(net.vs['political_party']))
        param_int = [pps.index(p) for p in net.vs['political_party']]
        vc = VertexClustering(net,param_int,params={'weight':net.es['weight']})
        mods_pps.append(vc.modularity)
        
        pps = list(set(net.vs['community']))
        param_int = [pps.index(p) for p in net.vs['community']]
        vc = VertexClustering(net,param_int,params={'weight':net.es['weight']})
        mods_comm.append(vc.modularity)
    
    plt.figure(figsize=(12,3))
    plt.plot(dates,mods_pps,'o',ls='-',label='political party')
    plt.plot(dates,mods_comm,'o',ls='-',label='community')
    plt.xticks(dates,rotation=45)
    plt.xlabel('year')
    plt.ylabel('modularity')
    plt.legend(loc='upper right')
    plt.savefig(header+'modularity.pdf',bbox_inches="tight")
    plt.close()

def shortest_path_mean(datas,nets):
    means = []
    for net in nets:
        vcount = net.vcount()
        dists = []
        for v in net.vs:
            path_lens = net.get_shortest_paths(v,to=net.vs,weights='weight')
            for p in path_lens:
                x = sum(net.es[idx]['distance'] for idx in p)
                if x > 0:
                    dists.append(x)
        m = mean(dists)
        means.append(m)
    dates = [int(d) for d in datas]
    plot(dates,means,'year','shortest paths mean','shortest_paths_mean')

def clustering_coefficient(datas,nets):
    clus_coefs = []
    for net in nets:
        clus = net.transitivity_local_undirected(weights=net.es['weight'])
        clus = np.nanmean(clus)
        
        clus_coefs.append(clus)
    dates = [int(d) for d in datas]
    plot(datas,clus_coefs,'year','clustering coefficient','clustering_coef')

def norm_mutual_info(dates,nets):
    mutual_infos = []
    for net in nets:
        comms = net.vs['community']
        pps = net.vs['political_party']
        
        mutual_info = normalized_mutual_info_score(comms,pps,'geometric')
        mutual_infos.append(mutual_info)
    
    dates2 = [int(d) for d in dates]
    plot(dates2,mutual_infos,'year','normalized mutual information','nmi')

def div_by_param(dates,nets,param):
    divs = []
    totals = []
    for date,net in zip(dates,nets):
        pps = net.vs[param]
        unique,count = np.unique(pps,return_counts=True)
        total = sum(count)
        probs = count/total
        entropy = -sum(np.log(probs)*probs)
        div = np.exp(entropy)
        divs.append(div)
        totals.append(len(unique))
    return divs,totals

def div(dates,nets):
    pps_div,pps_total = div_by_param(dates,nets,'political_party')
    comm_div,comm_total = div_by_param(dates,nets,'community')

    plt.figure(figsize=(12,2))
    plt.plot(dates,pps_div,label='political party diversity')
    plt.plot(dates,comm_div,label='community diversity')
    plt.plot(dates,pps_total,label='total of political parties')
    plt.plot(dates,comm_total,label='total of communities')
    plt.legend(loc='upper right',bbox_to_anchor=(1.2, 1.0))
    labels = [str(int(x)) if float(x)%5 == 0 else '' for x in dates]
    plt.xticks(np.arange(min(dates), max(dates)+1, 1.0),labels=labels)
    plt.xlabel('year')
    plt.savefig(header+'divs.pdf',format='pdf',bbox_inches="tight")
    plt.close()

if __name__ == '__main__':

    dates,nets = read_nets_by_years('data/1991-2019/by_year/dep*_0.8_leidenalg_dist.xnet')

    degree(dates,nets)
    modularity(dates,nets,'community','political_party')
    shortest_path_mean(dates,nets)
    dep_by_pp(dates,nets)
    clustering_coefficient(dates,nets)
    norm_mutual_info(dates,nets)
    div(dates,nets)
