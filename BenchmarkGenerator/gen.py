import numpy as np
import os
from BenchmarkGenerator import AStarSearchSolver as solver
from BenchmarkGenerator import utils
import collections as col

#** private functions
def __edge_traffic_stat(edge_traffic,gridSize):
    """
     edge_traffic_stat_ get statiscal information of edge traffic by solving problems
     with A* search    
    """
    via_capacity  = np.zeros((gridSize,gridSize))
    hoz_capacity = np.zeros((gridSize,gridSize-1)) # Only for Layer 1
    vet_capacity = np.zeros((gridSize-1,gridSize)) # Only for Layer 2
    for i in range(edge_traffic.shape[0]):
        connection = edge_traffic[i,:].astype(int)
#        print(connection)
        diff = (connection[3]-connection[0],\
                connection[4]-connection[1],\
                connection[5]-connection[2])
        if diff[0] == 1:
            hoz_capacity[connection[1],connection[0]] \
            = hoz_capacity[connection[1],connection[0]] + 1
        elif diff[0] == -1:
            hoz_capacity[connection[1],int(connection[0])-1] \
            = hoz_capacity[connection[1],int(connection[0])-1] + 1
        elif diff[1] == 1:
            vet_capacity[connection[1],connection[0]] \
            = vet_capacity[connection[1],connection[0]] + 1
        elif diff[1] == -1:
            vet_capacity[int(connection[1])-1,connection[0]] \
            = vet_capacity[int(connection[1])-1,connection[0]] + 1
        elif abs(diff[2]) == 1:
            via_capacity[connection[0],connection[1]] \
            = via_capacity[connection[0],connection[1]] + 1
        else:
            continue
    return via_capacity, hoz_capacity, vet_capacity


def __connection_statistical(edge_traffic,gridSize,benchmarkNumber):
    # get connection statistcal in vertical and horizontal direction, 
    # ignoring the via capacity since they tends to be large (set as 10 in simulated env)
    # cleaned edge traffic as input
    connection_cleaned = np.empty((0,6))
    for i in range(edge_traffic.shape[0]):
        connection = edge_traffic[i,:]
        if connection[3]<connection[0] or connection[4]<connection[1] or connection[5]<connection[2]:
            connection_flip = np.asarray([connection[3],connection[4],connection[5],\
                connection[0],connection[1],connection[2]])
            connection_cleaned = np.vstack((connection_cleaned,connection_flip))
        else:
            connection_cleaned = np.vstack((connection_cleaned,connection))

    connection_statistical = np.empty((0,7)) # last position is used for counting 
    connection_list = []

    for i in range(connection_cleaned.shape[0]):
        connectioni = connection_cleaned[i,:]
        # remove via connection before append to list
        if connectioni[2] == connectioni[5]:
            connection_list.append(tuple(connectioni))
    counter = col.Counter(connection_list)
    for key, value in counter.items():
        # print (key,value)
        statisticali = [int(i) for i in key]
        # Normalize the last column by benchmark numbers
        statisticali.append(int(value/benchmarkNumber))
        statisticali = np.asarray(statisticali)
        connection_statistical = np.vstack((connection_statistical,statisticali))
    # Sort connection_statistical
    connection_statistical_list = connection_statistical.tolist()
    connection_statistical_list.sort(key=lambda x: x[6])

    connection_statistical_array = np.asarray(connection_statistical_list)

    return connection_statistical_array


#***  part1  normal cap
def genBenchMark(benchmark_name,gridSize,netNum,vCap,hCap,maxPinNum):
    file = open(benchmark_name, 'w+')
    # Write general information
    file.write(f'grid {gridSize} {gridSize} 2\n')
    file.write(f'vertical capacity 0 {vCap}\n')
    file.write(f'horizontal capacity {hCap} 0\n')
    file.write('minimum width 1 1\n')
    file.write('minimum spacing 0 0\n')
    file.write('via spacing 0 0\n')
    file.write('0 0 10 10\n')
    file.write(f'num net {netNum}\n')
    # Write nets information 
    pinNum = np.random.randint(2,maxPinNum+1,netNum) # Generate Pin Number randomly
    for i in range(netNum):
        specificPinNum = pinNum[i]
        file.write(f'A{i+1} 0{i+1} {specificPinNum} 1\n')               #*  ex.  A1 01 3 1   #specificPinNum ==3
        xCoordArray = np.random.randint(1,10*gridSize,specificPinNum)   
        yCoordArray = np.random.randint(1,10*gridSize,specificPinNum)
        for j in range(specificPinNum):                                 ##19 30 1      #specificPinNum ==3
            file.write(f'{xCoordArray[j]}  {yCoordArray[j]} 1\n')       ## 12 15 1
    # Write capacity information                                        ## 29 51 1     
    file.write('0')
    file.close()
    return

def genSolution_genCapacity(vCap,gridSize,
                benchmarkdir, solutiondir,capacitydir):
    benEnum = 0;  edge_traffic =np.empty(shape=(0,6));
    for benchmarkfile in os.listdir(benchmarkdir):  #* pwd  src/benchmark   -> benchmarkfile = test_benchmark 1~N .gr
        benEnum = benEnum + 1
        edge_traffic_individual =np.empty(shape=(0,6))
        print('gen.py sol cap:',benchmarkdir+benchmarkfile,solutiondir)
        routeListMerged = solver.solve(benchmarkfile,solutiondir,benchmarkdir)   #! solving problems with A* search
        # print('routeListMerged',routeListMerged)
        for netCount in range(len(routeListMerged)):
            for pinCount in range(len(routeListMerged[netCount])-1):
                pinNow = routeListMerged[netCount][pinCount]
                pinNext = routeListMerged[netCount][pinCount+1]
                connection = [int(pinNow[3]),int(pinNow[4]),int(pinNow[2]),\
                              int(pinNext[3]),int(pinNext[4]),int(pinNext[2])]
                edge_traffic_individual = np.vstack((edge_traffic_individual,connection))
                edge_traffic = np.vstack((edge_traffic,connection))

        connection_statistical_array = __connection_statistical(edge_traffic_individual,gridSize,1)
        totalEdgeUtilized = connection_statistical_array.shape[0]
        edge_utilize_plot = vCap - connection_statistical_array[:,-1] # Assumption: vCap = hCap
        
        utils.plot2data(data1=edge_utilize_plot,label1="Capacity after A* route",
                        data2=vCap*np.ones_like(edge_utilize_plot),label2="Capacity before A* route",
                        y_label='Remaining capacity',
                        savepath=f'{capacitydir}edgePlotwithCapacity{vCap}number{benEnum}.png')

        # Draw heatmap of individual problems
        via_capacity_individual,hoz_capacity_individual,vet_capacity_individual =\
         __edge_traffic_stat(edge_traffic_individual,gridSize)
        
        utils.imshow(via_capacity_individual,'Via Capacity Heatmap',
                    f'{capacitydir}viaCapacity_{benchmarkfile}.png')
        utils.imshow(vet_capacity_individual,'Vertical Capacity Heatmap (Layer2)',
                    f'{capacitydir}vetCapacity_{benchmarkfile}.png')
        utils.imshow(hoz_capacity_individual,'Horizontal Capacity Heatmap (Layer1)',
                    f'{capacitydir}hozCapacity_{benchmarkfile}.png')
    # print("Total num of edge uitilization: ",edge_traffic.shape[0]) # print total num of edge uitilization   
    return edge_traffic

def cont_genCapacity(gridSize, benchmarkNumber,   edge_traffic,
                          capacitydir):
    via_capacity,hoz_capacity,vet_capacity = __edge_traffic_stat(edge_traffic,gridSize)
    connection_statistical_array = __connection_statistical(edge_traffic,gridSize,benchmarkNumber)
    totalEdgeUtilized = connection_statistical_array.shape[0]

     #draw a heat map of capacity utilization
    edgeNum = gridSize*gridSize + 2*gridSize*(gridSize-1)
    utilization_frequency = np.empty((0,2)) # 0 column: utilization times, 1 column: num of edges
    unsedEdge = [int(0),int(np.abs(edgeNum-totalEdgeUtilized))]
    utilization_frequency = np.vstack((utilization_frequency,unsedEdge))
    frequency_basket = []
    for i in range(connection_statistical_array.shape[0]):
        frequency_basket.append(int(connection_statistical_array[i,6]))
    counter_frequency = col.Counter(frequency_basket)
    for key, value in counter_frequency.items():
        frequencyi = [key,value] 
        utilization_frequency = np.vstack((utilization_frequency,frequencyi))

    utils.imshow(via_capacity,'Via Capacity Heatmap',
                 f'{capacitydir}viaCapacity.png')
    utils.imshow(vet_capacity,'Vertical Capacity Heatmap (Layer2)',
                 f'{capacitydir}vetCapacity.png')
    utils.imshow(hoz_capacity,'Horizontal Capacity Heatmap (Layer1)',
                 f'{capacitydir}hozCapacity.png')
    # bar plt
    utils.barplot(utilization_frequency,'Edge Utilization Histogram',
                  f'{capacitydir}edgeHist.png')
    return  connection_statistical_array

#**** part2  reduced cap
def generator_reducedCapacity(benchmark_name,gridSize,netNum,vCap,hCap,maxPinNum,\
        reducedCapNum,connection_statistical_array
    ):
    """Input parameters:
    Default setting: minWidth = 1, minSpacing = 0, 
    tileLength = 10, tileHeight = 10

    1. Grid Size: for 8-by-8 benchmark, grid size is 8
    2. vCap and hCap represents vertical capacity for layer2,
    and horizontal capacity for layer 1; unspecified capacity are 
    by default 0
    3. maxPinNum: maximum number of pins of one net
    4. capReduce: number of capacity reduction specification, by default = 0
    """
    # generate benchmarks with reduced capacity
    file = open('%s' % benchmark_name, 'w+')
    
    # Write general information
    file.write('grid {gridSize} {gridSize} 2\n'.format(gridSize=gridSize))
    file.write('vertical capacity 0 {vCap}\n'.format(vCap=vCap))
    file.write('horizontal capacity {hCap} 0\n'.format(hCap=hCap))
    file.write('minimum width 1 1\n')
    file.write('minimum spacing 0 0\n')
    file.write('via spacing 0 0\n')
    file.write('0 0 10 10\n')
    file.write('num net {netNum}\n'.format(netNum=netNum))
    # Write nets information 
    pinNum = np.random.randint(2,maxPinNum+1,netNum) # Generate Pin Number randomly
    for i in range(netNum):
        specificPinNum = pinNum[i]
        file.write('A{netInd} 0{netInd} {pin} 1\n'.format(netInd=i+1,pin=specificPinNum))
        xCoordArray = np.random.randint(1,10*gridSize,specificPinNum)
        yCoordArray = np.random.randint(1,10*gridSize,specificPinNum)
        for j in range(specificPinNum):
            file.write('{x}  {y} 1\n'.format(x=xCoordArray[j],y=yCoordArray[j]))
    # Write capacity information
    file.write('{capReduce}\n'.format(capReduce=reducedCapNum))
    for i in range(reducedCapNum):
        obstaclei = connection_statistical_array[-(i+1),0:6]
        file.write('{a} {b} {c}   {d} {e} {f}   3\n'.format(a=int(obstaclei[0]),b=int(obstaclei[1]),c=int(obstaclei[2]),\
         d=int(obstaclei[3]),e=int(obstaclei[4]),f=int(obstaclei[5])))
    file.close()
    return

def genSol_red_genCap_red(vCap,gridSize,
                            benchmarkdir, solutiondir,capacitydir):
    """# Get statistical information about edge traffic by solving benchmarks with A*Star Search"""
    edge_traffic =np.empty(shape=(0,6)) # initialize edge traffic basket #(data structure: startTile x,y,z and endTile x,y,z )
    benEnum = 0
    for benchmarkfile in os.listdir(benchmarkdir):
        benEnum = benEnum + 1
        edge_traffic_individual =np.empty(shape=(0,6))
        print('gen.py sol_red cap_red:',benchmarkdir+benchmarkfile,solutiondir)
        routeListMerged = solver.solve(benchmarkfile,solutiondir,benchmarkdir)   #* A* solver  # solving problems with A* search
        for netCount in range(len(routeListMerged)):
            for pinCount in range(len(routeListMerged[netCount])-1):
                pinNow = routeListMerged[netCount][pinCount]
                pinNext = routeListMerged[netCount][pinCount+1]
                connection = [int(pinNow[3]),int(pinNow[4]),int(pinNow[2]),\
                              int(pinNext[3]),int(pinNext[4]),int(pinNext[2])]
                edge_traffic_individual = np.vstack((edge_traffic_individual,connection))
                edge_traffic = np.vstack((edge_traffic,connection))
       
        connection_statistical_array = __connection_statistical(edge_traffic_individual,gridSize,1)
        totalEdgeUtilized = connection_statistical_array.shape[0]
        edge_utilize_plot = vCap - connection_statistical_array[:,-1] # Assumption: vCap = hCap
        utils.plot2data(data1=edge_utilize_plot,label1="Capacity after A* route",
                        data2=vCap*np.ones_like(edge_utilize_plot),label2="Capacity before A* route",
                        y_label='Remaining capacity',
                        savepath=f'{capacitydir}edgePlotwithCapacity{vCap}number{benEnum}.png')

        # Draw heatmap of individual problem
        via_capacity_individual,hoz_capacity_individual,vet_capacity_individual =\
         __edge_traffic_stat(edge_traffic_individual,gridSize)

        utils.imshow(via_capacity_individual,'Via Capacity Heatmap',
                     f'{capacitydir}viaCapacity_{benchmarkfile}.png')
        utils.imshow(vet_capacity_individual,'Vertical Capacity Heatmap (Layer2)',
                     f'{capacitydir}vetCapacity_{benchmarkfile}.png')
        utils.imshow(hoz_capacity_individual,'Horizontal Capacity Heatmap (Layer1)',
                     f'{capacitydir}hozCapacity_{benchmarkfile}.png')        
    # calculate capacity utilization
    # print("Total num of edge uitilization: ",edge_traffic.shape[0]) # print total num of edge uitilization
    return edge_traffic

def cont_gen_red_Capacity(gridSize, benchmarkNumber,   edge_traffic,
                         cap_red_dir):
    via_capacity,hoz_capacity,vet_capacity = __edge_traffic_stat(edge_traffic,gridSize)
    connection_statistical_array = __connection_statistical(edge_traffic,gridSize,benchmarkNumber)
    # print(connection_statistical_array[-30:-1,:])
    totalEdgeUtilized = connection_statistical_array.shape[0]
            # print('totalEdgeUtilized',totalEdgeUtilized)
        #    print(via_capacity)
        #    for i in range(via_capacity.shape[0]):
        #        for j in range(via_capacity.shape[1]):
        #            print(via_capacity[i,j])
        
     #draw a heat map of capacity utilization
    edgeNum = gridSize*gridSize + 2*gridSize*(gridSize-1)
    utilization_frequency = np.empty((0,2)) # 0 column: utilization times, 1 column: num of edges
    unsedEdge = [int(0),int(np.abs(edgeNum-totalEdgeUtilized))]
    utilization_frequency = np.vstack((utilization_frequency,unsedEdge))
    frequency_basket = []
    for i in range(connection_statistical_array.shape[0]):
        frequency_basket.append(int(connection_statistical_array[i,6]))
    counter_frequency = col.Counter(frequency_basket)
    for key, value in counter_frequency.items():
        frequencyi = [key,value] 
        utilization_frequency = np.vstack((utilization_frequency,frequencyi))
        # for key, value in counter.items():
        # # print (key,value)
        # statisticali = [int(i) for i in key]
        # statisticali.append(value)
        # statisticali = np.asarray(statisticali)
        # connection_statistical = np.vstack((connection_statistical,statisticali))
    # print(utilization_frequency)
    utils.imshow(via_capacity,'Via Capacity Heatmap',
                    f'{cap_red_dir}viaCapacity.png')
    utils.imshow(vet_capacity,'Vertical Capacity Heatmap (Layer2)',
                    f'{cap_red_dir}vetCapacity.png')
    utils.imshow(hoz_capacity,'Horizontal Capacity Heatmap (Layer1)',
                    f'{cap_red_dir}hozCapacity.png') 
    
    utils.barplot(utilization_frequency,'Edge Utilization Histogram',
                  f'{cap_red_dir}edgeHist.png',colorbar=False)

    return







