# Multiverse-Project

"Small World Brain Networks and General Intelligence: Investigating the Relationship across Various Analytical Pipelines"

Objective: To investigate the correlation between the general factor of intelligence ('g') and small-world brain network topology using 72 distinct analytical pipelines and to identify clusters of pipelines that exhibited comparable correlation patterns.

Methods: Whole-brain resting-state functional magnetic resonance imaging (fMRI) data of 827 participants was obtained from the Human Connectome Brain project. The data was processed using a carefully guided multiverse of analytical pipelines. Subsequently, small-worldness values were derived and examined for their correlation with the respective 'g' scores of the participants.

Results: Findings revealed little to no associations between the general factor of intelligence ('g') and small-world brain network topology across the 72 distinct analytical pipelines. The projection of relationships in lower-dimensional space exhibited minimal variability between pipelines, challenging previous positive findings.

# **About**
- [MultiverseCode.py]: This Python file contains all the functions that were created for each choice in the multiverse. This code runs for one subject because looping over all subjects for each analytical choice was more confusing in terms of data management and computationally expensive. It was easier to create it for one subject and run it parallelly for 90 subjects in the high-performance cluster ROSA.
  
- [MultiverseCode.job]: This is a bash script created to run the Python file as a job in the HPC environment. It can be used to set various parameters according to the user's choice, for instance, the total amount of time the code should be allowed to run or the number of jobs that can run in parallel.

- [setup_venv.sh]: This is another shell script to make sure all the relevant packages were already installed in the HPC environment before running the job.

- The code on this repository was run using Python 3.10.8 and on the HPC Cluster ROSA, located at the University of Oldenburg (Germany) and funded by the DFG through its Major Research Instrumentation Programme (INST 184/225-1 FUGG) and the Ministry of Science and Culture (MWK) of the Lower Saxony State.

**You can refer to the [Workflow.md] file to get an understanding of how the code is structured.**


