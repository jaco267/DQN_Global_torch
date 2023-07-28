import numpy as np
import os
import matplotlib.pyplot as plt
import collections as col

from BenchmarkGenerator import gen
np.random.seed(1)
def make_many_dir(dir_list):
    for dir_ in dir_list:
        os.makedirs(dir_,exist_ok=True)
    return

def main_fn(benchmarkNumber=10,    #* 100
            gridSize = 8,  
            netNum =  20,   
            vCap = 4,hCap = 4,   
            maxPinNum = 5,  
            reducedCapNum = 3, 
            prefix = '../train_data_/'
    ):
    os.system(f'rm -r {prefix}')# Make sure previous benchmark files: "benchmark","capacityplot", and 'solution' are removed

    benchmarkdir = f'{prefix}benchmark/'; capacitydir = f'{prefix}capacityPlot_A*/'; solutiondir = f'{prefix}solutionsA*/'; 
    benchmark_red_dir = f'{prefix}benchmark_reduced/'; cap_red_dir = f'{prefix}capacityPlot_A*_reduced/'; sol_red_dir = f'{prefix}solutionsA*_reduced/';
    
    make_many_dir([benchmarkdir,capacitydir,solutiondir,
                    benchmark_red_dir,cap_red_dir,sol_red_dir])

    print("-----part1  benchmark with normal capacity-----")
    for i in range(benchmarkNumber):                        #?  benchmark
        gen.genBenchMark(f"{benchmarkdir}test_benchmark_{i+1}.gr",gridSize,netNum,vCap,hCap,maxPinNum)
    print("Benchmark gen done")
    edge_traffic = gen.genSolution_genCapacity(             #?  capacityPlot_A*  solutionsA*
        vCap,gridSize, benchmarkdir, solutiondir,capacitydir)   #* solving problems with A* search
    connection_statistical_array = gen.cont_genCapacity(    #?  capacityPlot_A*
        gridSize, benchmarkNumber, edge_traffic,  capacitydir)  # calculate capacity utilization
    print("-----part2  benchmark with reduced capacity-----")
    for i in range(benchmarkNumber):                        #!  benchmark_reduced
        gen.generator_reducedCapacity(f"{benchmark_red_dir}test_benchmark_{i+1}.gr",gridSize,netNum,
            vCap,hCap,maxPinNum, reducedCapNum,connection_statistical_array)
    edge_traffic = gen.genSol_red_genCap_red(vCap,gridSize, #!  capacityPlot_A*_reduced/  solutionsA*_reduced/ 
                            benchmark_red_dir, sol_red_dir,cap_red_dir)
    gen.cont_gen_red_Capacity(gridSize,                     #!  capacityPlot_A*_reduced/ 
              benchmarkNumber, edge_traffic,  cap_red_dir)
    return








