"""
Evaluator for the function minimization example
"""

import re
import subprocess
import time
import traceback
import random

def run_with_timeout(program_path, timeout_seconds=60):
    """
    Run a function with a timeout using subprocess.

    Args:
        func: Function to run
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        timeout_seconds: Timeout in seconds

    Returns:
        Result of the function or raises TimeoutError
    """
    cmd_compile = ["nvcc", "-o", "mtxTranspose", program_path]
    try:
        # Run the command and grab its output using subprocess.Popen
        proc = subprocess.Popen(cmd_compile, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        #proc = subprocess.Popen(cmd_compile, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
        exit_code = proc.returncode
        if exit_code != 0:
            print(stderr)  # Print the error output if the command failed
            raise RuntimeError(f"Process exited with code {exit_code}")
    except subprocess.TimeoutExpired:
        # Kill the process if it times out
        proc.kill()
        raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    #nrow = random.randint(11,20)
    #ncol = random.randint(11,20)
    nrow = 16384
    ncol = 16381
    cmd_run = ["./mtxTranspose", str(nrow), str(ncol)]
    try:
        # Run the command and grab its output using subprocess.Popen
        proc = subprocess.Popen(cmd_run, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        #proc = subprocess.Popen(cmd_run, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
        exit_code = proc.returncode
        if exit_code != 0:
            print(stderr)  # Print the error output if the command failed
            raise RuntimeError(f"Process exited with code {exit_code}")
    except subprocess.TimeoutExpired:
        # Kill the process if it times out
        proc.kill()
        raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    # expected output 
    # Input matrix (3x4):
    # 0 1 2 3 
    # 4 5 6 7 
    # 8 9 10 11 
    #
    # Kernel execution time: 234.988 ms
    # Effective memory bandwidth: 3.80475e-07 GB/s
    # Output (transposed) matrix:
    # 0 4 8 
    # 1 5 9 
    # 2 6 10 
    # 3 7 11 
    pattern = (
        r"Kernel execution time:\s*([-+]?\d*\.\d+|\d+) ms\n"
        r"Effective memory bandwidth:\s*([-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+) GB/s"
    )
    match = re.search(pattern, stdout)
    if not match:
        raise ValueError("Expected summary lines not found")

    exec_time, mem_bdw = map(float, match.groups())
    return exec_time, mem_bdw


def evaluate(program_path):
    """
    Evaluate the program by compile then run the binary and fetching performance metrics.

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    try:
        # For constructor-based approaches, a single evaluation is sufficient
        # since the result is deterministic
        start_time = time.time()

        # Use subprocess to compile then run with timeout
        exec_time, mem_bdw = run_with_timeout(
            program_path, timeout_seconds=60  # Single timeout
        )

        end_time = time.time()
        eval_time = end_time - start_time

        # Combined score - higher is better
        #combined_score = correct / total if total > 0 else 0.0

        print(
            f"Kernel Execution Time: ={exec_time} ms, Memory Bandwidth={mem_bdw} GB/sec, Evaluation Time={eval_time} sec"
        )

        return {
            "exec_time": exec_time,
            "mem_bdw": mem_bdw,
            "eval_time": eval_time,
        }

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        traceback.print_exc()
        return {
            "exec_time": 0.0,
            "mem_bdw": 0.0,
            "eval_time": 0.0,
        }
