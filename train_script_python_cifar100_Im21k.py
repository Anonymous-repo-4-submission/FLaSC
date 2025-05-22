import subprocess
import re
import math
import os
import logging
import sys

def run_command(command):
    """Runs a shell command and returns its output."""
    result = subprocess.run(command, capture_output=True)
    output = result.stdout.decode('utf-8').strip() 
    floats = re.findall(r'[-+]?\d*\.\d+|\d+', output)
    return float(floats[-1])

def extract_result(output):
    """Extracts the result from the output using regex."""
    match = re.search(r'Result:\s*([0-9]+(\.[0-9]+)?)', output)
    if match:
        return float(match.group(1))
    return None

def calculate_stats(results):
    """Calculates and returns the mean, variance, and standard deviation."""
    count = len(results)
    sum_values = sum(results)
    sum_sq = sum(value ** 2 for value in results)

    mean = sum_values / count
    variance = (sum_sq / count) - (mean ** 2)
    stddev = math.sqrt(variance)

    return mean, variance, stddev

# Loop over shot values
method = "FLaSC"
logs_name = "./FLaSC"
if not os.path.exists(logs_name):
        os.makedirs(logs_name)
logfilename = "./FLaSC/{}".format(method)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(filename)s] => %(message)s",
    handlers=[
        logging.FileHandler(filename=logfilename + ".log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logging.info('Starting_{}'.format(method))
for shot in range(1,2):
    results_cmd1 = []
    results_cmd2 = []

    # Loop over seed values
    for seed in range(0,10):
        print(f"Running with seed: {seed} and shot: {shot}")
        command1 = [
            "bash", "-c",
            f"CUDA_VISIBLE_DEVICES=0 python main.py -i 1 -d cifar224 "
            f"-seed_path ./subsets/cifar100/shot{shot}_seed{seed}.txt -class_samples_seed {seed} -shot {shot}"
        ]
        # breakpoint()
        value_cmd1 = run_command(command1)
       
        if value_cmd1 is not None:
            print(f"valid result found with seed {seed} and shot {shot} is {value_cmd1}")
            logging.info('valid result found with seed {} and shot {} is {}'.format(seed,shot,value_cmd1))
            results_cmd1.append(value_cmd1)
        else:
            print(f"No valid result found with seed {seed} and shot {shot}")
            logging.info('No valid result found with seed {} and shot {}'.format(seed,shot))

    print(f"Statistics for Our method Im21k, Shot {shot}:")
    logging.info('Statistics for Our method Im21k, Shot {}: '.format(shot))
    if results_cmd1:
        mean, variance, stddev = calculate_stats(results_cmd1)
        print(f"Mean: {mean}")
        print(f"Variance: {variance}")
        print(f"Standard Deviation: {stddev}")
        logging.info('Mean {}: '.format(mean))
        logging.info('Variance {}: '.format(variance))
        logging.info('Standard Deviation {}: '.format(stddev))
        logging.info('Done... ')

    else:
        print(f"No results to calculate statistics for Shot {shot}.")

