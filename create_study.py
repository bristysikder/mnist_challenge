import json, sys, copy, getpass
from itertools import product

from madry_tools.client.rpc_client import RPCClient
from madry_tools.core import datatypes

### STUDY-LEVEL CONFIGURATION SETTINGS

# a commit to a version of this repository with a stable version of the main
# script
GIT_COMMIT = 'https://github.com/MadryLab/adversarial_cifar/commit/0d92e07c24c3d3180f1eb447269615f2e41d27d5'
MAIN_FILE_NAME = 'nat_train.py'
FILES_TO_ARCHIVE = ['job_parameters.json', 'config.py', 'job_result.json']

### CREATE STUDY

# establish connection to production database; use port=50052 for test mode
client = RPCClient()
username = getpass.getuser()

study_name = 'MNIST_Compression'
study_id, error_msg = client.create_study(study_name)

if error_msg is not None:
    print(error_msg)
    sys.exit(1)

train_steps = [i * 5000 for i in xrange(200)]


### CREATE JOBS
jobs = []
prio = 200
k = 10
for t in train_steps:
    params = {'max_num_training_steps': t}
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