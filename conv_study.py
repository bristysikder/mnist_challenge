import json, sys, copy, getpass
from itertools import product

from madry_tools.client.rpc_client import RPCClient
from madry_tools.core import datatypes

### STUDY-LEVEL CONFIGURATION SETTINGS

# a commit to a version of this repository with a stable version of the main
# script
GIT_COMMIT = 'https://github.com/bristysikder/mnist_challenge/commit/d9345e5fe97251692a4682acd8358779cfe50802'
MAIN_FILE_NAME = 'conv_train.py'
FILES_TO_ARCHIVE = ['job_parameters.json']

### CREATE STUDY

# establish connection to production database; use port=50052 for test mode
client = RPCClient()
username = getpass.getuser()

study_name = 'mnist_conv_v5'
study_id, error_msg = client.create_study(study_name)

if error_msg is not None:
    print(error_msg)
    sys.exit(1)

# Epsilon from Algorith #1 of Sanjeev Aroras paper
epsilons = [0.5, 0.25, 0.1, 0.05, 0.01, 0.005]
first_fc = [True, False]
second_fc = [True, False]
train_steps = [ 1000, 10000, 100000, 1000000] 
grid = product(epsilons, first_fc, second_fc, train_steps) 

### CREATE JOBS
jobs = []
prio = 200

for eps, f, s, steps  in grid:
    model_dir = ('models/compression_prio_%d') % prio
    params = {'c_eps': eps, 
 	      'nu': eps, 
              'model_dir': model_dir,
              'compress_first_fc' : f,
	      'compress_second_fc' : s, 
              'max_num_training_steps' : steps}
    parameter_json = json.dumps(params)
    job_info = datatypes.JobCreationInfo(
        git_commit=GIT_COMMIT,
        main_file_name=MAIN_FILE_NAME,
        files_to_archive=FILES_TO_ARCHIVE,
        parameter_json=parameter_json,
        priority=float(prio))
    prio -= 1
    jobs.append(job_info)

### ADD JOBS TO STUDY
error_msg = client.add_jobs(study_id, jobs)
if error_msg is not None:
    print(error_msg)
