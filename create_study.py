import json, sys, copy, getpass
from itertools import product

from madry_tools.client.rpc_client import RPCClient
from madry_tools.core import datatypes

### STUDY-LEVEL CONFIGURATION SETTINGS

# a commit to a version of this repository with a stable version of the main
# script
GIT_COMMIT = 'https://github.com/bristysikder/mnist_challenge/commit/81e75b62736608239670598fe4060e0a29042905'
MAIN_FILE_NAME = 'nat_train.py'
FILES_TO_ARCHIVE = ['job_parameters.json']

### CREATE STUDY

# establish connection to production database; use port=50052 for test mode
client = RPCClient()
username = getpass.getuser()

study_name = 'mnist_compression_simple_mlp'
study_id, error_msg = client.create_study(study_name)

if error_msg is not None:
    print(error_msg)
    sys.exit(1)

# Epsilon from Algorith #1 of Sanjeev Aroras paper
epsilons = [0.5, 0.25, 0.1, 0.05]

### CREATE JOBS
jobs = []
prio = 200

for i, eps in enumerate( epsilons):
    model_dir = ('models/compression_%d') % i
    params = {'c_eps': eps, 'nu': eps, 
              'model_dir': model_dir }
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
