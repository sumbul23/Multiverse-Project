#import base64

#import numpy as np
#from networkx import random_reference
#from scipy import signal
import pickle
import numpy as np
import pandas as pd
import pywt
import random
#import matplotlib.pyplot as plt
import networkx as nx

from scipy import signal
#from scipy.signal import butter, filtfilt

#from pymatreader import read_mat

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import io
#import plotly.graph_objects as go
import sys

import os
os.chdir('/dss/work/nola7251')
from pathlib import Path
PROJECT_ROOT = Path.cwd()
data_path = PROJECT_ROOT / 'Dataset'
output_path = PROJECT_ROOT / 'Output_new'
if not data_path.is_dir():
    data_path.mkdir(parents=True)

wgsr = os.listdir(data_path/'fMRI-wgsr')
gsr = os.listdir(data_path/'MRI-gsr')

# Finding elements in one list that are not in the other
wgsr_not_in_gsr = [name for name in wgsr if name[:-11] + '.mat' not in gsr]
gsr_not_in_wgsr = [name for name in gsr if name[:-4] + '_ROI_ts.mat' not in wgsr]

datanames_wgsr = [item for item in wgsr if item not in wgsr_not_in_gsr]
datanames_wgsr = sorted(datanames_wgsr)

datanames_gsr = [item for item in gsr if item not in gsr_not_in_wgsr]
datanames_gsr = sorted(datanames_gsr)

#Creating final dataset that matches with g_scores
#Load csv file and get the subject ID list
df = pd.read_csv('/dss/work/nola7251/score_factor.csv')
subject_ID = df['ID_subj'].tolist()

# Find elements in (fmri-wgsr & mri-gsr) that are not in subject_ID(scores)

wgsr_not_in_scores = [name for name in datanames_wgsr if int(name[:-11]) not in subject_ID]
gsr_not_in_scores = [name for name in datanames_gsr if int(name[:-4]) not in subject_ID]

final_wgsr = [item for item in datanames_wgsr if item not in wgsr_not_in_scores]
final_wgsr = sorted(final_wgsr)

final_gsr = [item for item in datanames_gsr if item not in gsr_not_in_scores]
final_gsr = sorted(final_gsr)

# Initialize data structures for each subfolder
data_gsr = []
data_wgsr = []

for data_name in final_wgsr:
    file_path = os.path.join(data_path, 'fMRI-wgsr', data_name)
    data = io.loadmat(file_path)
    data_wgsr.append(data)

for data_name in final_gsr:
    file_path = os.path.join(data_path, 'MRI-gsr', data_name)
    data = io.loadmat(file_path)
    data_gsr.append(data)

#This is how the subject will be input and printed when running on the HPC cluster.

num_subjects = np.size(data_gsr)
subject = int(sys.argv[1])-1
print("You entered:", subject)

def low_pass_filter(datal):
    b, a = signal.butter(8, 0.087)  # 0.125 * nyquist freq
    z, p, k = signal.tf2zpk(b, a)
    eps = 1e-9
    r = np.max(np.abs(p))
    approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))

    lpf = np.empty((np.size(datal)-2, datal[0][0].shape[0], datal[0][0].shape[1]))
    
    #lpfimpulse is optional: only created for visualization
    lpfimpulse = np.empty((np.size(datal)-2, datal[0][0].shape[0], datal[0][0].shape[1]))

#There was data from 4 sessions. run in [0,1] will take data from session 1 and 2, whereas run in [2,3] will take data from session 3 and 4.

    for run in [0,1]:#range(np.size(datal)):
        x_values = datal[0][run]

        for region in range(x_values.shape[0]):
            y1 = signal.filtfilt(b, a, x_values[region, :], method='gust')
            lpf[run, region, :] = y1

            y2 = signal.filtfilt(b, a, x_values[region, :], method='gust', irlen=approx_impulse_len)
            lpfimpulse[run, region, :] = y2

            #print(np.max(np.abs(y1 - y2)))

    return lpf, lpfimpulse

#You can comment out lpfimpulse in the function and here if you dont want those values.
lpf_gsr, lpfimpulse_gsr = low_pass_filter(data_gsr[subject]['ROI_ts'])
lpf_wgsr, lpfimpulse_wgsr = low_pass_filter(data_wgsr[subject]['ROI_ts'])


def band_pass_filter(datab):
    # Define sampling rate and filter parameters
    sampling_rate = 1 / 0.72
    low_freq = 0.01
    high_freq = 0.1

    # Define filter order
    nyquist_freq = 0.5 * sampling_rate
    filter_order = 3
    low_cutoff = low_freq / nyquist_freq
    high_cutoff = high_freq / nyquist_freq

    # Create band-pass filter
    bp, ap = signal.butter(filter_order, [low_cutoff, high_cutoff], btype='band')

    bpf = np.empty((np.size(datab)-2, datab[0][0].shape[0], datab[0][0].shape[1]))

#Necessary to change [0,1] to [2,3] here if you want data from session 3 and 4. 

    for run in [0,1]:#range(np.size(datab)):
        x2_values = datab[0][run]

        for region in range(x2_values.shape[0]):
            filtered_data = signal.filtfilt(bp, ap, x2_values[region, :], axis=-1)
            bpf[run, region, :] = filtered_data

    return bpf


bpf_gsr = band_pass_filter(data_gsr[subject]['ROI_ts'])
bpf_wgsr = band_pass_filter(data_wgsr[subject]['ROI_ts'])


def calculate_functional_connectivity(datafc):
    num_runs = datafc.shape[0]
    num_regions = datafc.shape[1]

    df_pearson = np.empty((num_runs, num_regions, num_regions))
    df_partial = np.empty((num_runs, num_regions, num_regions))
    #df_wavelet = [[] for _ in range(num_runs)]


    for run in range(datafc.shape[0]):
        x3_values = datafc[run]

        # Compute Pearson correlation matrix
        pearson_corr = np.corrcoef(x3_values)
        df_pearson[run] = pd.DataFrame(pearson_corr)

        # Compute partial correlation matrix
        ts_con = x3_values.T
        ic = np.linalg.inv(np.cov(ts_con, rowvar=False))
        diag_ic = np.sqrt(np.abs(np.diag(ic)))
        r_corr = ic / np.outer(diag_ic, diag_ic)
        np.fill_diagonal(r_corr, 0.0)
        df_partial[run] = pd.DataFrame(r_corr)

        # Compute wavelet correlation matrix using discrete wavelet transform
        # wavelet_corr = pywt.wavedecn(x3_values, 'db4', level=1, mode='periodization')
        # df_wavelet[run].append(wavelet_corr)

    return df_pearson, df_partial #, df_wavelet

Pearsonfc_gsr_lp, Partialfc_gsr_lp = calculate_functional_connectivity(lpf_gsr)
Pearsonfc_gsr_bp, Partialfc_gsr_bp =  calculate_functional_connectivity(bpf_gsr)
Pearsonfc_wgsr_lp, Partialfc_wgsr_lp =  calculate_functional_connectivity(lpf_wgsr)
Pearsonfc_wgsr_bp, Partialfc_wgsr_bp =  calculate_functional_connectivity(bpf_wgsr)


# For calculating continuous wavelet transform. Note that data should be a 360x1200 matrix

def calculate_wavelet_coherence(datafc):

    num_runs = datafc.shape[0]
    num_regions = datafc.shape[1]
    time_series = datafc.shape[2]

    df_wavelet = [[] for _ in range(num_runs)]

    # Define the scales and frequencies for the wavelet transform
    scales = np.arange(1, 128)
    frequencies = 1 / scales
    for run in range(datafc.shape[0]):
        x3_values = datafc[run]

        # Perform the continuous wavelet transform for each brain area
        cwt_result = np.zeros((x3_values.shape[0], len(scales), x3_values.shape[1]), dtype=complex)
        for region in range(x3_values.shape[0]):
            cwt_result[region, :, :] = pywt.cwt(x3_values[region, :], scales, 'morl')[0]

        # Calculate wavelet coherence between brain areas
        def wavelet_coherence(a, b):

            # Calculate the cross-wavelet transform
            xwt_ab = cwt_result[a, :, :] * np.conj(cwt_result[b, :, :])

            # Calculate the wavelet power spectrum
            xwt_aa = np.abs(cwt_result[a, :, :]) ** 2
            xwt_bb = np.abs(cwt_result[b, :, :]) ** 2

            # Calculate wavelet coherence
            coherence = np.abs(np.sum(xwt_ab, axis=1) / np.sqrt(np.sum(xwt_aa, axis=1) * np.sum(xwt_bb, axis=1)))
            average_coherence = np.mean(coherence)
            return average_coherence

        # Calculate wavelet coherence for all pairs of brain areas
        coherence_matrix = np.zeros((num_regions, num_regions))
        for i in range(num_regions):
            for j in range(num_regions):
                coherence_mat = wavelet_coherence(i, j)
                coherence_matrix[i, j] = coherence_mat
        df_wavelet[run] = coherence_matrix
        # Display the coherence matrix or perform further analysis
        # plt.imshow(coherence_matrix, cmap='viridis', aspect='auto')
        # plt.colorbar()
        # plt.title('Wavelet Coherence' + str(run))
        # plt.show()

    return df_wavelet



Waveletfc_gsr_lp = calculate_wavelet_coherence(lpf_gsr)
Waveletfc_gsr_bp = calculate_wavelet_coherence(bpf_gsr)
Waveletfc_wgsr_lp = calculate_wavelet_coherence(lpf_wgsr)
Waveletfc_wgsr_bp = calculate_wavelet_coherence(bpf_wgsr)

#To save the results obtained so far in a dictionary
cmx = [Partialfc_gsr_bp, Partialfc_gsr_lp, Partialfc_wgsr_bp,Partialfc_wgsr_lp, Pearsonfc_gsr_bp, Pearsonfc_gsr_lp, Pearsonfc_wgsr_bp, Pearsonfc_wgsr_lp]

cmx = {'Partialfc_gsr_bp': Partialfc_gsr_bp, 'Partialfc_gsr_lp': Partialfc_gsr_lp, 'Partialfc_wgsr_bp': Partialfc_wgsr_bp, 'Partialfc_wgsr_lp': Partialfc_wgsr_lp, 'Pearsonfc_gsr_bp': Pearsonfc_gsr_bp, 'Pearsonfc_gsr_lp': Pearsonfc_gsr_lp, 'Pearsonfc_wgsr_bp': Pearsonfc_wgsr_bp, 'Pearsonfc_wgsr_lp': Pearsonfc_wgsr_lp}

cmx_wav =  {'Waveletfc_gsr_bp': Waveletfc_gsr_bp, 'Waveletfc_gsr_lp': Waveletfc_gsr_lp, 'Waveletfc_wgsr_bp': Waveletfc_wgsr_bp, 'Waveletfc_wgsr_lp': Waveletfc_wgsr_lp}

#cmx_wav['Waveletfc_gsr_bp'][run][list0][list/dict at 0/1]['da' - only when dict].shape

# for key, value in cmx.items():
#     print(key, value.shape)

def create_binary_graph(adj_matrix):

    G = nx.Graph()
    num_nodes = adj_matrix.shape[0]

    # Add nodes to the graph
    G.add_nodes_from(range(num_nodes))

    # Add edges to the graph
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i, j] > 0:
                G.add_edge(i, j)

    return G

def create_weighted_graph(adj_matrix):

    G = nx.Graph()
    num_nodes = adj_matrix.shape[0]

    # Add nodes to the graph
    G.add_nodes_from(range(num_nodes))

    # Add edges to the graph
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i, j] > 0:
                G.add_edge(i, j, weight=abs(adj_matrix[i,j]))

    return G

binary_graphs = []
weighted_graphs = []

#Looping for all 4 runs

for key, adj_matrix in cmx.items():
    print(key)
    for i in range(2):
        bg = create_binary_graph(adj_matrix[i])
        binary_graphs.append(bg)
        wg = create_weighted_graph(adj_matrix[i])
        weighted_graphs.append(wg)


def create_binary_graph_from_wavelet_coeffs(coeffs):

    G = nx.Graph()
    num_nodes = coeffs.shape[0]

    # Add nodes to the graph
    G.add_nodes_from(range(num_nodes))

    #Add edges to the graph
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if coeffs[i, j] > 0:
                G.add_edge(i, j)

    return G

def create_weighted_graph_from_wavelet_coeffs(coeffs):

    G = nx.Graph()
    num_nodes = coeffs.shape[0]

    # Add nodes to the graph
    G.add_nodes_from(range(num_nodes))

    # Add edges to the graph
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if coeffs[i, j] > 0:
                G.add_edge(i, j, weight=abs(coeffs[i, j]))

    return G


binary_graphs_wav = []
weighted_graphs_wav = []

for key, wav_mat in cmx_wav.items():
    print(key)
    for i in range(2):
        run_wav = wav_mat[i]
        print(run_wav)
        bg_wav = create_binary_graph_from_wavelet_coeffs(run_wav)
        binary_graphs_wav.append(bg_wav)

        wg_wav = create_weighted_graph_from_wavelet_coeffs(run_wav)
        weighted_graphs_wav.append(wg_wav)

weighted_graphs.extend(weighted_graphs_wav)
binary_graphs.extend(binary_graphs_wav)


#Sigma of Wavelet graphs will work with the nx.sigma(wav_graph, niter=2, nrand=2, seed=None)


def relative_threshold(graph, threshold_fraction):

    first_edge = next(iter(graph.edges(data=True)), None)
    if first_edge is not None and 'weight' in first_edge[2]:
        print("The graph is weighted.")
        sorted_edges = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    else:
        print("The graph is binary.")
        sorted_edges = list(graph.edges())

    # Determine the number of edges to keep based on the threshold fraction
    num_edges_to_keep = int(len(sorted_edges) * threshold_fraction)

    # Keep only the top fraction of edges
    thresholded_edges = sorted_edges[:num_edges_to_keep]

    # Create a new graph with the thresholded edges
    thresholded_graph = nx.Graph()
    thresholded_graph.add_edges_from(thresholded_edges)

    return thresholded_graph


threshold_fractions = [0.1, 0.3, 0.5]

threshold_structures = {}

for fraction in threshold_fractions:
    binary_thresholded_graphs = [relative_threshold(g, fraction) for g in binary_graphs ]

    weighted_thresholded_graphs = [relative_threshold(g, fraction) for g in weighted_graphs]

    threshold_structures[f'Threshold_{fraction}'] = {
        'binary': binary_thresholded_graphs,
        'weighted': weighted_thresholded_graphs
    }

#Important because networkx functions don't work for unconnected graphs.
def check_connectedness(all_graphs):
    not_connected_graphs = {}

    for threshold, graph_types in all_graphs.items():
        not_connected_graphs[threshold] = {}

        for graph_type, graphs in graph_types.items():
            not_connected_graphs[threshold][graph_type] = []

            for idx, graph in enumerate(graphs):
                if not nx.is_connected(graph):
                    not_connected_graphs[threshold][graph_type].append(idx)

                    # Unfrozen version of the graph
                    unfrozen_graph = graph.copy(as_view=False)

                    connected_components = list(nx.connected_components(unfrozen_graph))
                    largest_component_nodes = max(connected_components, key=len)
                    largest_connected_graph = unfrozen_graph.subgraph(largest_component_nodes)

                    # Replace the original graph with the largest connected component
                    all_graphs[threshold][graph_type][idx] = largest_connected_graph

    return not_connected_graphs, all_graphs


not_connected_graphs, modified_structure = check_connectedness(threshold_structures)



#A manual function was created for calculating small worldness and small world propensity because the Networkx sigma and omega functions took a very long time. Only in cases where there was an error, did we calculate the values using networkx functions.


def calculate_sigma_and_phi(all_graphs):

    #global random_clustering, random_path_length
    num_randomizations = 2
    sigmaphi_binary_graphs = []
    sigmaphi_weighted_graphs = []

    # for threshold, graph_types in all_graphs.items():
    #     print(f"Threshold: {threshold}")
    for graph_type, graphs in all_graphs.items():
        print(f"Graph Type: {graph_type}")
        if graph_type == 'binary':
            for idx, graph in enumerate(graphs):
                print(f"Graph {idx + 1}:")
                #Calculate clustering coefficient and shortest path length
                clustering_coefficient = nx.average_clustering(graph)
                short_path_length = nx.average_shortest_path_length(graph)

                #Unfrozen version of the graph for double edge swapping
                unfrozen_graph = graph.copy(as_view=False)
                
                # Calculate random network measures
                num_nodes = graph.number_of_nodes()
                num_edges = graph.number_of_edges()
                # For sigma:

                random_graphs = []
                for _ in range(num_randomizations):
                    try:
                        random_graph = nx.double_edge_swap(unfrozen_graph, nswap=2, max_tries=num_edges*10)
                        random_graphs.append(random_graph)
                    except nx.NetworkXAlgorithmError:
                        continue
                if random_graphs:
                    print("Sigma function worked")
                    # For sigma: Calculate the clustering coefficient and characteristic path length of the random graphs
                    random_clustering = np.mean([nx.average_clustering(g) for g in random_graphs])
                    random_path_length = np.mean([nx.average_shortest_path_length(g) for g in random_graphs])
                    # Calculate sigma using the random clustering and random path length
                    sig = (clustering_coefficient / random_clustering) / (short_path_length / random_path_length)
                    print(sig)

                else:
                    # if there is a networkx error for sigma
                    # Calculate sigma using the nx.sigma function
                    sig = nx.sigma(unfrozen_graph, niter=2, nrand=2, seed=None)
                    print("nx.sigma worked", sig)

                #For phi:
                if random_graphs:
                    #For phi:Create lattice graph
                    lattice_graph = nx.watts_strogatz_graph(num_nodes, int(num_edges / num_nodes), 0)

                    # For phi: Calculate the clustering coefficient and characteristic path length of the lattice and random graphs
                    C_lattice = nx.average_clustering(lattice_graph)
                    L_lattice = nx.average_shortest_path_length(lattice_graph)

                    C_random = random_clustering
                    L_random = random_path_length
                    # For phi: Assigning the clustering coefficient and characteristic path length of the graph
                    C_brain = clustering_coefficient
                    L_brain = short_path_length

                    # For phi: Calculate the deviations
                    delta_C = (C_lattice - C_brain) / (C_lattice - C_random)
                    delta_L = (L_brain - L_random) / (L_lattice - L_random)

                    # Calculate the small-world propensity
                    phi = 1 - np.sqrt((delta_C**2 + delta_L**2) / 2)
                    print("Phi function worked", phi)
                else:
                    phi = nx.omega(unfrozen_graph, niter=2, nrand=2, seed=None)
                    print("nx.omega worked", phi)

                sigmaphi_binary_graphs.append((graph_type, graph, {'sigma': sig}, {'phi': phi}, {'cc': clustering_coefficient}, {'path_len': short_path_length}))

        elif graph_type == 'weighted':
            for idx, graph in enumerate(graphs):
                print(f"Graph {idx + 1}:")
                clustering_coefficient = nx.average_clustering(graph, weight='weight')
                short_path_length = nx.average_shortest_path_length(graph, weight='weight')
                num_nodes = graph.number_of_nodes()
                num_edges = graph.number_of_edges()
                #Unfrozen copy of graph
                unfrozen_graph = graph.copy(as_view=False)

                random_graphs = []
                for _ in range(num_randomizations):
                    try:
                        random_graph = nx.double_edge_swap(unfrozen_graph, nswap=2, max_tries=num_edges * 10)
                        random_graphs.append(random_graph)
                    except nx.NetworkXAlgorithmError:
                        continue
                if random_graphs:
                    print("Sigma function worked")
                    random_weighted_clustering = np.mean([nx.average_clustering(g, weight='weight') for g in random_graphs])
                    random_weighted_path_length = np.mean([nx.average_shortest_path_length(g, weight='weight') for g in random_graphs])
                    sig = (clustering_coefficient / random_weighted_clustering) / (short_path_length / random_weighted_path_length)
                    print(sig)
                else:
                    # Calculate sigma using the nx.sigma function
                    sig = nx.sigma(unfrozen_graph, niter=2, nrand=2, seed=None)
                    print("nx.sigma worked", sig)

                #For phi:
                if random_graphs:
                    lattice_graph = nx.watts_strogatz_graph(num_nodes, int(num_edges / num_nodes), 0)
                    for edge in lattice_graph.edges():
                        lattice_graph[edge[0]][edge[1]]['weight'] = random.uniform(0, 1)

                    # Calculate the clustering coefficient and characteristic path length of the lattice and random graphs
                    C_lattice = nx.average_clustering(lattice_graph, weight='weight')
                    C_random = random_weighted_clustering
                    L_lattice = nx.average_shortest_path_length(lattice_graph, weight='weight')
                    L_random = random_weighted_path_length

                    # Assigning the clustering coefficient and characteristic path length of the graph
                    C_brain = clustering_coefficient
                    L_brain = short_path_length

                    # Calculate the deviations
                    delta_C = (C_lattice - C_brain) / (C_lattice - C_random)
                    delta_L = (L_brain - L_random) / (L_lattice - L_random)

                    # Calculate the small-world propensity

                    phi = 1 - np.sqrt((delta_C**2 + delta_L**2) / 2)
                    print("Phi function worked", phi)
                else:
                    phi = nx.omega(unfrozen_graph, niter=2, nrand=2, seed=None)
                    print("nx.omega worked", phi)
                # # Update the existing tuple to include the sigma values
                sigmaphi_weighted_graphs.append((graph_type, graph, {'sigma': sig}, {'phi': phi},  {'cc': clustering_coefficient}, {'path_len': short_path_length}))


    return sigmaphi_binary_graphs, sigmaphi_weighted_graphs

sigmaphi_binary_graphs_10, sigmaphi_weighted_graphs_10 = calculate_sigma_and_phi(modified_structure['Threshold_0.1'])
sigmaphi_binary_graphs_30, sigmaphi_weighted_graphs_30 = calculate_sigma_and_phi(modified_structure['Threshold_0.3'])
sigmaphi_binary_graphs_50, sigmaphi_weighted_graphs_50 = calculate_sigma_and_phi(modified_structure['Threshold_0.5'])

sigmaphi_binary = [sigmaphi_binary_graphs_10, sigmaphi_binary_graphs_30, sigmaphi_binary_graphs_50]
sigmaphi_weighted = [sigmaphi_weighted_graphs_10, sigmaphi_weighted_graphs_30, sigmaphi_weighted_graphs_50]


# Define a list of data types
data_types = ['cmx', 'cmx_wav', 'binary', 'weighted']

# Create separate directories for each data type if they don't exist
for data_type in data_types:
    data_type_directory = os.path.join(output_path, data_type)
    os.makedirs(data_type_directory, exist_ok=True)

# Load or generate your data for the current subject and data types
cmx_data = cmx
cmx_wav_data = cmx_wav
binary_data = sigmaphi_binary
weighted_data = sigmaphi_weighted

ID = final_gsr[subject]

# Save the data for the current subject into the relevant data type folders
with open(os.path.join(output_path, 'cmx', f'{ID}_cmx.pkl'), 'wb') as pkl_file:
    pickle.dump(cmx_data, pkl_file)
with open(os.path.join(output_path, 'cmx_wav', f'{ID}_cmx_wav.pkl'), 'wb') as pkl_file:
    pickle.dump(cmx_wav_data, pkl_file)
with open(os.path.join(output_path, 'binary', f'{ID}_binary.pkl'), 'wb') as pkl_file:
    pickle.dump(binary_data, pkl_file)
with open(os.path.join(output_path, 'weighted', f'{ID}_weighted.pkl'), 'wb') as pkl_file:
    pickle.dump(weighted_data, pkl_file)


# op = '/home/jafri/mount_dir/Output'
# # # Load the data from the binary file using pickle
# with open(os.path.join(op, 'binary', '100307.mat_binary.pkl'), 'rb') as pkl_file:
#     loaded_datab = pickle.load(pkl_file)
# with open(os.path.join(op, 'cmx', '100307.mat_cmx.pkl'), 'rb' )as pkl_file:
#     loaded_cmx = pickle.load(pkl_file)
#
# with open(os.path.join(op, 'weighted','100307.mat_weighted.pkl'), 'rb') as pkl_file:
#     loaded_dataw = pickle.load(pkl_file)
