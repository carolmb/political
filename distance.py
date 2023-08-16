import xnet
import glob
import math
import matplotlib
import concurrent.futures

import numpy as np
import matplotlib.pyplot as plt

from igraph import *
from collections import defaultdict

from util import get_valid_pp
from util import filter_pp_name

def calculate_dist(filenames):
	for filename in filenames:
		# print(filename)
		net = xnet.xnet2igraph(filename)
		weights = net.es['weight']
		weights = [math.sqrt(2*(1-w)) for w in weights]
		if len(weights) > 0:
			net.es['distance'] = weights
			xnet.igraph2xnet(net,filename[:-5]+"_dist.xnet")
		else:
			print('error',filename)

def to_sort(dates,nets):
    dates = np.asarray(dates)
    nets = np.asarray(nets)

    sorted_idxs = np.argsort(dates)
    dates = dates[sorted_idxs]
    nets = nets[sorted_idxs]
    return dates,nets

# Utilidades
def get_freqs(summaries,dates):
    ys = defaultdict(lambda:defaultdict(lambda:[]))

    freq_dict = defaultdict(lambda:[])
    for d in dates:
        year_summary = summaries[d]
        for pp1,summary_pp1 in year_summary.items():
            if summary_pp1:
                for pp2,(mean,std,f) in summary_pp1.items():
                    ys[pp1][pp2].append((d,mean,std,f))
                    freq_dict[pp2].append(f)

    freq = [(np.nanmean(fs),pp) for pp,fs in freq_dict.items()]
    freq = sorted(freq,reverse=True)
    
    i = 0
    f_max = freq[i][0]
    while np.isnan(freq[i][0]):
        i+= 1
        f_max = freq[i][0]
    
    return ys,freq,f_max

def plot_metric(to_plot,interval_colors,color,output_fname,metric_name,is_custom_labels,is_bg):
    
    plt.figure(figsize=(12,3))
    
    xs2 = []
    
    print(output_fname)

    for pp1,(means,total_std,fraq,xs) in to_plot.items():
        if len(xs) > len(xs2):
            xs2 = xs
        fraq = max(fraq,0.45)
#         elw = max(0.3,2*fraq)
#         lw = max(0.3,2*fraq)
#         ms = max(0.3,2*fraq)
        
        plt.errorbar(xs,means,total_std,
        linestyle='-',label=pp1.upper(),fmt='o',elinewidth=1.5*fraq,
        linewidth=2*fraq,markersize=2*fraq,
        alpha=max(0.6,fraq),color=color[pp1])     
    
    delta = 12
    if is_custom_labels:
        delta = 1

    xpos = np.arange(min(xs2), max(xs2)+1/delta, 1/delta)
    labels = [str(int(x)) if i%delta == 0 else '' for i,x in enumerate(xpos)]
    
    
    plt.xticks(xpos,labels=labels,rotation=35)   
    
    if is_bg:
        for begin,delta,color in interval_colors:
            if begin+delta >= xs2[0] and begin <= xs2[-1]:
                plt.axvspan(max(begin,xs2[0]), min(begin+delta,xs2[-1]), facecolor=color, alpha=0.3)
                plt.axvline(max(begin,xs2[0]),color='#2e2e2e',linestyle='--',alpha=0.5)

    plt.legend(loc='upper right',bbox_to_anchor=(1.05, 1.0))
    plt.xlabel('year')
    plt.ylabel(metric_name)                
    plt.savefig(output_fname+'.pdf',format='pdf',bbox_inches="tight")
    plt.clf()


# Menores caminhos
def calculate_shortest_paths(net,pps):
    summary = defaultdict(lambda:defaultdict(lambda:0))
    all_paths = []
    
    for pp1 in pps:
        sources = net.vs.select(political_party_eq=pp1)
        for pp2 in pps:
#             print('current pps:',pp1,pp2)
            targets = net.vs.select(political_party_eq=pp2)
            targets = [v.index for v in targets]
            paths = []
#             for s in sources: 
#                 for t in targets:
#                     print(net.shortest_paths_dijkstra(source=s,target=t,weights='distance')[0],end=',')
            for s in sources:
                path_lens = net.get_shortest_paths(s,to=targets,weights='distance',output="epath")
                for p in path_lens:
                    x = sum(net.es[idx]['distance'] for idx in p)
#                     print(x,end=',')
                    if x > 0:
                        paths.append(x)
                        all_paths.append(x)
            if len(paths) == 0:
                summary[pp1][pp2] = (np.nan,np.nan,np.nan)
                summary[pp2][pp1] = (np.nan,np.nan,np.nan)
            else:
                mean = np.mean(paths)
                std_dev = np.std(paths)
                summary[pp1][pp2] = (mean,std_dev,len(targets)) 
                summary[pp2][pp1] = (mean,std_dev,len(sources))

            if pp1 == pp2:
                break
    all_paths_mean = np.mean(all_paths)
    all_paths_std = np.std(all_paths)
    return summary,(all_paths_mean,all_paths_std)

def shortest_path_by_pp(freq,pp2_means,f_max):
    to_plot = dict()
    for f,pp2 in freq:
        means_std = pp2_means[pp2]
        means_std = np.asarray(means_std)
        means = means_std[:,1]
        std = means_std[:,2]
        xs = means_std[:,0]
        fraq = f/f_max

        if not np.isnan(means).all():
            to_plot[pp2] = (means,std,fraq,xs)
    return to_plot

def plot_shortest_paths(dates,nets,valid_pps,interval_colors,color,header,is_custom_labels,is_bg):

    summaries = dict()
    all_paths_summary = []
    for date,net in zip(dates,nets):
        summaries[date],all_paths = calculate_shortest_paths(net,valid_pps)
        all_paths_summary.append(all_paths)
    all_paths_summary = np.asarray(all_paths_summary)
    
    ys,_,_ = get_freqs(summaries,dates)
    
    for pp1,pp2_means in ys.items():
        if not pp1 in valid_pps:
            continue
        freq = []
        for pp2,means_std in pp2_means.items():
            means_std = np.array(means_std)
            freq.append((np.nanmean(means_std[:,3]),pp2))
        freq = sorted(freq,reverse=True)
        f_max = freq[0][0]
        
        to_plot = shortest_path_by_pp(freq,pp2_means,f_max)
        to_plot['all'] = (all_paths_summary[:,0], all_paths_summary[:,1],0.3,dates)
        
        plot_metric(to_plot,interval_colors,color,header+pp1,'average shortest path len',is_custom_labels,is_bg)

def plot_shortest_paths_all_years(dates,nets,valid_pps,interval_colors,color,is_bg):
    header = 'shortest_path_'
    plot_shortest_paths(dates,nets,valid_pps,interval_colors,color,header,True,is_bg)
    
def plot_shortest_paths_mandate(dates,nets,year,valid_pps,interval_colors,color,is_bg):
    idxs = [idx for idx,date in enumerate(dates) if date < year+4 and date >= year]
    current_dates = [dates[idx] for idx in idxs]
    current_nets = [nets[idx] for idx in idxs]

    header = 'shortest_path_' + str(year) + '_' + str(year+3) + '_'
    
    plot_shortest_paths(current_dates,current_nets,valid_pps,interval_colors,color,header,False,is_bg)

# Isolamento/Fragmentação
def fragmentation_to_plot(summaries,dates):
    to_plot = dict()
    
    ys,freq,f_max = get_freqs(summaries,dates)
    
    fragmentation = dict()
    for f,pp1 in freq:
        pp2_means = ys[pp1]
        
        means = np.zeros(len(pp2_means[pp1]))
        xs = []
        for pp2,means_std in pp2_means.items():
            if pp1 == pp2:
                means_std = np.array(means_std)
                means = means_std[:,1]
                std = means_std[:,2]
                xs = means_std[:,0]
                break
        fraq = f/f_max
        fraq = max(fraq,0.45)
        if np.isnan(fraq) or np.isnan(means).all():
            continue
        to_plot[pp1] = (means,std,fraq,xs)
    
    return to_plot

def isolation_to_plot(summaries,dates):
    to_plot = dict()
    
    ys,freq,f_max = get_freqs(summaries,dates)
#     if np.isnan(f_max):
#         return None,None
    for f,pp1 in freq:
        pp2_means = ys[pp1]
#         if pp1 == 'psl':
#             print(pp2_means)
            
        means = np.zeros(len(pp2_means[pp1]))
        total_std = np.zeros(len(pp2_means[pp1]))
        total = np.zeros(len(pp2_means[pp1]))
        xs = []
        for pp2,means_std in pp2_means.items():
            if not pp1 == pp2:
                means_std = np.array(means_std)
                means_std[np.isnan(means_std)]=0
                if not np.isnan(means_std).any():
                    xs = means_std[:,0]
                    t = means_std[:,3]
                    std = means_std[:,2]
                    total += t
                    means += means_std[:,1]*t
                    total_std += std*t

        means /= total
        total_std /= total
        fraq = f/f_max
        fraq = max(fraq,0.45)
        if np.isnan(fraq) or np.isnan(means).all():
            continue
        to_plot[pp1] = (means,total_std,fraq,xs)
    return to_plot

def plot_metric_all_years(dates,nets,metric_to_plot,valid_pps,pps_color,metric_name,is_bg):
    summaries = dict()

    for d,n in zip(dates,nets):
        summaries[d],all_paths = calculate_shortest_paths(n,valid_pps)

    output_fname = metric_name + '_' + str(min(dates))+'_'+str(max(dates))

    to_plot = metric_to_plot(summaries,dates)
    metric = {pp1:(means,total_std) for pp1,(means,total_std,_,_) in to_plot.items()}
    plot_metric(to_plot,interval_colors,pps_color,output_fname,metric_name,True,is_bg)
    
    return metric,dates

def plot_metric_mandate(dates,nets,metric_to_plot,year,valid_pps,pps_color,metric_name,is_bg,delta=4):
    summaries = dict()

    idxs = [idx for idx,date in enumerate(dates) if date < year+delta and date >= year]
     
    current_dates = [dates[idx] for idx in idxs]
    current_nets = [nets[idx] for idx in idxs]

    for d,n in zip(current_dates,current_nets):
        summaries[d],all_paths = calculate_shortest_paths(n,valid_pps)

    output_fname = metric_name + '_' + str(int(min(current_dates)))+'_'+str(int(max(current_dates)))

    to_plot = metric_to_plot(summaries,current_dates)
    metric = {pp1:(means,total_std) for pp1,(means,total_std,_,_) in to_plot.items()}
    plot_metric(to_plot,interval_colors,pps_color,output_fname,metric_name,False,is_bg)
    
    return metric,current_dates



if __name__ == '__main__':
    ##############################################################
    # READ INPUT
    ##############################################################
    source_by_year = 'data/1991-2019/by_year/dep_*_obstr_0.8_leidenalg'
    source_by_mandate = 'data/1991-2019/mandate/dep_*_0.8'

    # Called only once
    source = 'data/1991-2019/by_year/dep_*_obstr_0.8_leidenalg'
    filenames = glob.glob(source+'.xnet')
    calculate_dist(filenames)

    filenames_by_year = sorted(glob.glob(source_by_year+'_dist.xnet'))
    filenames_by_mandate = sorted(glob.glob(source_by_mandate+'_dist.xnet'))

    dates_by_year, dates_by_mandate = [],[]
    nets_by_year, nets_by_mandate = [],[]

    for filename in filenames_by_year:
        net = xnet.xnet2igraph(filename)
        net.vs['political_party'] = [filter_pp_name(p) for p in net.vs['political_party']]
        nets_by_year.append(net.components().giant())

        date = int(filename.split('dep_')[1].split('_')[0])
        dates_by_year.append(date)
        
    for filename in filenames_by_mandate:
        net = xnet.xnet2igraph(filename)
        net.vs['political_party'] = [filter_pp_name(p) for p in net.vs['political_party']]
        nets_by_mandate.append(net.components().giant())

        # por ano
        date = int(filename.split('dep_')[1].split('_')[0])
        date += float(filename.split('dep_')[1].split('_')[1])/12
        dates_by_mandate.append(date)

    dates_by_year,nets_by_year = to_sort(dates_by_year,nets_by_year)
    dates_by_mandate,nets_by_mandate = to_sort(dates_by_mandate,nets_by_mandate)

    ##############################################################
    # VALID POLITICAL PARTIES
    ##############################################################

    # valid_pps = list(get_valid_pp(nets_by_year,1990,1,cut_percent=0.06))
    # valid_pps = ['psdb', 'pp', 'pmdb', 'pt', 'dem', 'pl', 'ptb', 'psb', 'pr']
    # valid_pps = sorted(valid_pps)
    # valid_pps = ['psdb', 'pp', 'pmdb', 'pt', 'dem', 'pdt', 'psb', 'psl', 'ptb', 'prb', 'pl']
    valid_pps = ['psdb', 'pp', 'pmdb', 'pt', 'dem']#,'psl']

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] + ['magenta','navy','violet','teal']

    pps_color = dict()
    for pp,c in zip(valid_pps,colors):
        pps_color[pp] = c
    pps_color['all'] = 'cyan'
        
    interval_colors = [(1992.95,2.05,pps_color['pmdb']),(1995,8,pps_color['psdb']),
    (2003,13.4,pps_color['pt']),(2016.4,0.26,'#757373'),(2016.66,2.34,pps_color['pmdb'])]
    # ,
    # (2019,1,pps_color['psl'])] # psl

    govs = [('FHC',(1995.01,2003)),('Lula',(2003.01,2011)),('Dilma',(2011.01,2016.4)),('Temer',(2016.4,2019)),('Bolsonaro',(2019.1,2020))]
    gov_map = {'FHC':'psdb','Lula':'pt','Dilma':'pt','Temer':'pmdb','Bolsonaro':'psl'}

    ##############################################################
    # PLOT SHORTEST PATHS
    ##############################################################

    # todos os anos
    plot_shortest_paths_all_years(dates_by_year,nets_by_year,valid_pps,interval_colors,pps_color,True)

    # por mandato
    for year in range(2016,2020,4):
        plot_shortest_paths_mandate(dates_by_mandate,nets_by_mandate,year,valid_pps,interval_colors,pps_color,False)

    ##############################################################
    # ISOLATION/FRAGMENTATION
    ##############################################################

    # Código para dados em intervalos de anos:
    plot_metric_all_years(dates_by_year,nets_by_year,isolation_to_plot,valid_pps,pps_color,'isolation',True)
    plot_metric_all_years(dates_by_year,nets_by_year,fragmentation_to_plot,valid_pps,pps_color,'fragmentation',True)


    # Código para dados em intervalos de meses:
    plot_metric_mandate(dates_by_mandate,nets_by_mandate,fragmentation_to_plot,2015,valid_pps,pps_color,'fragmentation',True,5)
    plot_metric_mandate(dates_by_mandate,nets_by_mandate,isolation_to_plot,2015,valid_pps,pps_color,'isolation',True,5)

    ##############################################################
    # ZOOM 2015 - 2020
    ##############################################################

    total_frag = defaultdict(lambda:[])
    total_xs = []
    for year in range(2015,2020,4):
        frag,xs = plot_metric_mandate(dates_by_mandate,nets_by_mandate,fragmentation_to_plot,year,valid_pps,pps_color,'fragmentation',True)
        for k,v in frag.items():
            total_frag[k].append(v)
        total_xs.append(xs)

    total_isol = defaultdict(lambda:[])
    total_xs = []
    for year in range(2015,2020,4):
        isol,xs = plot_metric_mandate(dates_by_mandate,nets_by_mandate,isolation_to_plot,year,valid_pps,pps_color,'isolation',True)
        for k,v in isol.items():
            total_isol[k].append(v)
        total_xs.append(xs)