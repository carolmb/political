import json
import zipfile
import unidecode

input_file_zip = "data/deputadosData_1991-2019.zip"
input_file = "data/deputadosData_1991-2019.json"
output_file = "data/deputadosv2.json"

with zipfile.ZipFile(input_file_zip, 'r') as zip_ref:
    zip_ref.extractall('data/')

file = open(input_file, 'r').read()
data = json.loads(file)

def name_filtering(name):
	name = name.lower()
	name = name.strip()
	name = unidecode.unidecode(name)
	return name

def get_mpv(nome, info, p):
	o = dict()
	o['nome'] = nome
	o['data_apresentacao'] = info['DataApresentacao']
	o['tema'] = info['tema']
	o['partido_autor'] = info['partidoAutor']
	o['objetivo'] = p['@ObjVotacao']
	o['data'] = p['@Data']
	o['votos'] = []
	deputies = p['votos']['Deputado']
	names = set()
	for d in deputies:
		dep = dict()
		dep_name = name_filtering(d['@Nome'])
		if dep_name in names:
			continue
		names.add(dep_name)
		dep['nome'] = dep_name
		dep['uf'] = d['@UF'].strip()
		dep['voto'] = d['@Voto'].strip()
		dep['partido'] = d['@Partido'].strip().lower()
		dep['id_deputado'] = d['@ideCadastro'].strip()
		o['votos'].append(dep)
	o['resumo'] = p['@Resumo']
	o['id'] = p['@codSessao']
	return o

output = dict()
output['proposicoes'] = []
output_set = set()
for year, prop in data.items():
	for nome, info in prop.items():
		if not 'VOTES' in info:
			continue
		polls = info['VOTES']['Votacoes']['Votacao']
		
		if type(polls) == type([]):
			for p in polls:
				o = get_mpv(nome, info, p)
				if not str(o) in output_set:
					output['proposicoes'].append(o)
					output_set.add(str(o))
		else:
			o = get_mpv(nome, info, polls)
			if not str(o) in output_set:
				output['proposicoes'].append(o)
				output_set.add(str(o))

output = json.dumps(output, indent=4, sort_keys=True)
file = open(output_file, 'w')
file.write(output)