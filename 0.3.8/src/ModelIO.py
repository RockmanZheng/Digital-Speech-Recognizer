import xml.etree.ElementTree as ET
import numpy as np
import pdb

# Read and write model related file
# Note that
# log sparse matrix: must be list consists of 3 lists, representing row_idx,
# col_idx, and log_val, respectively.
# log sparse 3d array: must be list consists of 4 lists, representing row_idx,
# col_idx, lay_idx, and log_val, respectively.

# Read in model file
# Get NAME, SCALES, STATE, LOGTRANS, LOGCOEF, MEAN, LOGVAR

def ReadModel(model_file):
    tree = ET.parse(model_file)
    model = tree.getroot()
    name = model.get('name')
    states = model.find('states').text.split()
    scales = model.find('scales')
    num_states = int(scales.find('num_states').text)
    num_components = int(scales.find('num_components').text)
    dim_observation = int(scales.find('dim_observation').text)
    log_trans_node = model.find('log_trans')
    log_trans = LogTransNodeParser(log_trans_node)
    log_coef_node = model.find('log_coef')
    log_coef = MatrixNodeParser(log_coef_node, num_states, num_components)
    mean_node = model.find('mean')
    mean = Array3DNodeParser(mean_node, num_states,
                             num_components, dim_observation)
    log_var_node = model.find('log_var')
    log_var = Array3DNodeParser(log_var_node, num_states, num_components, dim_observation)
    return name, states, num_states, num_components, dim_observation, log_trans, log_coef, mean, log_var

# Read in a list of model files

def ReadModels(model_files):
    names = []
    states_list = []
    num_states_list = []
    num_components_list = []
    dim_observation_list = []
    log_trans_list = []
    log_coef_list = []
    mean_list = []
    log_var_list = []
    for model_file in model_files:
        name, states, num_states, num_components, dim_observation, log_trans, log_coef, mean, log_var = ReadModel(model_file)
        names.append(name)
        states_list.append(states)
        num_states_list.append(num_states)
        num_components_list.append(num_components)
        dim_observation_list.append(dim_observation)
        log_trans_list.append(log_trans)
        log_coef_list.append(log_coef)
        mean_list.append(mean)
        log_var_list.append(log_var)
    return names, states_list, num_states_list, num_components_list, dim_observation_list, log_trans_list, log_coef_list, mean_list, log_var_list


# Read in pre-model file
# Get NAME, STATE, LOGTRANS
def ReadPreModel(pre_model_file):
    tree = ET.parse(pre_model_file)
    pre_model = tree.getroot()
    name = pre_model.get('Name')
    states = pre_model.find('States').text.split()
    num_states = int(pre_model.find('States-Number').text)
    nnz = int(pre_model.find('Non-Zero-Number').text)
    row_idx = list(map(int, pre_model.find('Row-Index').text.strip().split()))
    col_idx = list(map(int, pre_model.find('Colomn-Index').text.strip().split()))
    network = [row_idx, col_idx]
    return name, states, num_states, nnz, network

# Write model into file
# Iname: string
# Inum_states,Inum_components,Inum_observations,Idim_observation: int.  N, M,
# T, L, resp.
# Ilog_trans: list consists of 3 lists, int, int, float, resp.
# Ilog_coef: list consists of N lists, each of which is again a list of
# dimension M
# Imean: list matrix of dimension N*M, each entry is a list of dimension L
# Ilog_var: list matrix of dimension N*M, each entry is a list of dimension L
# model_file: file to write in

def WriteModel(Iname, Istates, Inum_states, Inum_components, Idim_observation, Ilog_trans, Ilog_coef, Imean, Ilog_var, model_file):
    model = ET.Element('model')
    model.set('name', Iname)
    states = ET.SubElement(model, 'states')
    states.text = ''
    for state in Istates:
        states.text += state + ' '
    scales = ET.SubElement(model, 'scales')
    num_states = ET.SubElement(scales, 'num_states')
    num_states.text = str(Inum_states)
    num_components = ET.SubElement(scales, 'num_components')
    num_components.text = str(Inum_components)
    dim_observation = ET.SubElement(scales, 'dim_observation')
    dim_observation.text = str(Idim_observation)
    log_trans = ET.SubElement(model, 'log_trans')
    LogTransNodeWriter(log_trans, Ilog_trans)
    log_coef = ET.SubElement(model, 'log_coef')
    MatrixNodeWriter(log_coef, Ilog_coef)
    mean = ET.SubElement(model, 'mean')
    Array3DNodeWriter(mean, Imean)
    log_var = ET.SubElement(model, 'log_var')
    Array3DNodeWriter(log_var, Ilog_var)
    model_tree = ET.ElementTree(model)
    model_tree.write(model_file)
    return

# Write pre-model into file

def WritePreModel(Iname, Istates, Inum_states, Inetwork, pre_model_file):
    pre_model = ET.Element('Pre-Model')
    pre_model.set('Name', Iname)
    states = ET.SubElement(pre_model, 'States')
    states.text = ''
    for state in Istates:
        states.text += state + ' '
    num_states = ET.SubElement(pre_model, 'States-Number')
    num_states.text = str(Inum_states)
    nnz = len(Inetwork[0])
    nnz_node = ET.SubElement(pre_model, 'Non-Zero-Number')
    nnz_node.text = str(nnz)
    row_idx = ET.SubElement(pre_model, 'Row-Index')
    col_idx = ET.SubElement(pre_model, 'Colomn-Index')
    row_idx.text = ''
    col_idx.text = ''
    for k in range(nnz):
        row_idx.text += str(Inetwork[0][k]) + ' '
        col_idx.text += str(Inetwork[1][k]) + ' '
    pre_model_tree = ET.ElementTree(pre_model)
    pre_model_tree.write(pre_model_file)
    return

# log_trans node parser

def LogTransNodeParser(log_trans_node):
    log_trans_row_idx = log_trans_node.find('row_idx').text.split()
    log_trans_row_idx = list(map(int, log_trans_row_idx))
    log_trans_col_idx = log_trans_node.find('col_idx').text.split()
    log_trans_col_idx = list(map(int, log_trans_col_idx))
    log_trans_log_val = log_trans_node.find('log_val').text.split()
    log_trans_log_val = list(map(float, log_trans_log_val))
    log_trans = [log_trans_row_idx, log_trans_col_idx, log_trans_log_val]
    return log_trans

# log_trans node writer

def LogTransNodeWriter(log_trans_node, log_trans):
    log_trans_row_idx = ET.SubElement(log_trans_node, 'row_idx')
    log_trans_row_idx.text = ''
    for idx in log_trans[0]:
        log_trans_row_idx.text += str(idx) + ' '
    log_trans_col_idx = ET.SubElement(log_trans_node, 'col_idx')
    log_trans_col_idx.text = ''
    for idx in log_trans[1]:
        log_trans_col_idx.text += str(idx) + ' '
    log_trans_log_val = ET.SubElement(log_trans_node, 'log_val')
    log_trans_log_val.text = ''
    for val in log_trans[2]:
        log_trans_log_val.text += str(val) + ' '
    return

# matrix node parser

def MatrixNodeParser(node, N, M):
    raw_matrix = node.text.split()
    raw_matrix = list(map(float, raw_matrix))
    matrix = []
    for i in range(N):
        row = []
        for m in range(M):
            row.append(raw_matrix[i * M + m])
        matrix.append(row)
    return matrix

# matrix node writer

def MatrixNodeWriter(node, matrix):
    node.text = ''
    for row in matrix:
        for entry in row:
            node.text += str(entry) + ' '
    return

# 3d array node parser

def Array3DNodeParser(node, N, M, L):
    raw_array = node.text.split()
    raw_array = list(map(float, raw_array))
    array3d = []
    for i in range(N):
        row = []
        for m in range(M):
            col = []
            for lay in range(L):
                col.append(raw_array[i * M * L + m * L + lay])
            row.append(col)
        array3d.append(row)
    return array3d

# 3d array node writer

def Array3DNodeWriter(node, array3d):
    node.text = ''
    for row in array3d:
        for col in row:
            for entry in col:
                node.text += str(entry) + ' '
    return

def LoadModels(model_id,MODEL_FOLDER):
	models = []
	for k in range(len(model_id)):
		# Load model
		model_filename = MODEL_FOLDER + model_id[k] + '.xml'
		name,states,num_states,num_components,dim_observation,log_trans,log_coef,mean,log_var = ReadModel(model_filename)
		model = [log_trans,log_coef,mean,log_var]
		models.append(model)
	return models