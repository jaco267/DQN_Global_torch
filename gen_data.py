import os
import argparse
import BenchmarkGenerator

import fire  

def parse_arguments():
    parser = argparse.ArgumentParser('Benchmark Generator Parser')
    parser.add_argument('--benchNumber',type=int,dest="benchmarkNumber",\
        help='number of problems in the experiment',default=20)
    parser.add_argument('--gridSize',type=int,dest='gridSize',
                        help='two layers routing area(w==h==gridSize)',default=16)
    parser.add_argument('--netNum',type=int,dest='netNum',help='number of nets that will needs to be routed',default=5)
    parser.add_argument('--capacity',type=int,dest='cap',help='edge capacity for problem (check paper for details)',default=4)
    parser.add_argument('--maxPinNum',type=int,dest='maxPinNum',help='''max number of pins in a net, the number of pins 
      													in one net follows a uniform distribution between [0,maxPinNum]''',default=5)
    parser.add_argument('--reducedCapNum',type=int,dest='reducedCapNum',help='''number of edges that has 
      reduced capacity (blolcked or partially blocked, check problem genrator part of the code 
      for details and make modification to your needs: e.g., do you want to block edges randomly 
      or just block high congestion area)''',default=1)   #todo  let the network learn by itself, reduce these hand-crafted method

    return parser.parse_args()
def main(benchmarkNumber=5,  #100??           #number of problems in the experiment  
	 		gridSize = 8,                      #two layers routing area(w==h==gridSize)
            netNum =  20,   				   #number of nets that will needs to be routed
            vCap = 4,hCap = 4,                 #edge capacity for problem (check paper for details)
            maxPinNum = 5,          #max number of pins in a net, the number of pins 
      													#in one net follows a uniform distribution between [0,maxPinNum]
            reducedCapNum = 3,     #number of edges that has reduced capacity (blolcked or partially blocked, check problem genrator part of the code 
   						 #  for details and make modification to your needs: e.g., do you want to block edges randomly  or just block high congestion area)
            prefix = './train_data_/'   #save folder
	):
	print(benchmarkNumber, gridSize ,  netNum , vCap ,hCap , maxPinNum ,  reducedCapNum ,  prefix )
	BenchmarkGenerator.main_fn(benchmarkNumber,    #* 100
            gridSize,    netNum,   
            vCap,hCap,   
            maxPinNum, reducedCapNum,  prefix
	)
if __name__ == '__main__':
	print('**************')
	print('Problem Generating Module')
	print('**************')
    #* generate a bunch of folder
	#** benchmark, benchmark_reduced  ,capacityPlot_A*,capacityPlot_A*_reduced,
	#**  solutionsA*,solutionsA*_reduced)

	fire.Fire(main)     #https://google.github.io/python-fire/guide/

	

