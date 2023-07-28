import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib import cm


class VisualGraph(object):
    def __init__(self,gridParameters):
        self.gridParameters = gridParameters
        return
    def show_grid(self):
        layer_num = self.gridParameters['gridSize'][2]   #2 in [8,8,2]  plot 2 layer
        board_w = self.gridParameters['gridSize'][0]  #8
        board_h = self.gridParameters['gridSize'][1]  #8
        fig, axes = plt.subplots(1,layer_num)  
        tile_width = self.gridParameters['tileWidth']     #10
        tile_height = self.gridParameters['tileHeight']   #10   #heigh in a grid
        for i in range(layer_num):
            x = np.random.random((board_w,board_h))  # print(x)  #just some random shade
            axes[i].imshow(x,origin='lower',
                           extent=(0,tile_width*board_w,0,tile_height*board_h),
                           alpha=0.7,cmap=cm.gray)
            axes[i].set(title=f'Layer {i}')

            # Visualize capacity
            if self.gridParameters['verticalCapacity'][i] == 0:
                for k in range(self.gridParameters['gridSize'][1]-1):
                    for j in range(self.gridParameters['gridSize'][0]):
                        rect = patches.Rectangle((0.2*tile_width+tile_width*j,0.9*tile_height+tile_height*k),0.6*tile_width,0.2*tile_height,facecolor='b')
                        axes[i].add_patch(rect)
            if self.gridParameters['horizontalCapacity'][i] == 0:
                for k in range(self.gridParameters['gridSize'][1] ):
                    for j in range(self.gridParameters['gridSize'][0]-1):
                        rect = patches.Rectangle((0.9*tile_width +tile_width*j,0.2*tile_height + tile_height*k),0.2*tile_width,0.6*tile_height, facecolor='b')
                        axes[i].add_patch(rect)
                #           fitst map   A1   
                """
                    ['17', '62', '1'],     # pin'1'
                    ['69', '30', '1'],    # pin'2'
                    ['22', '45', '1'],    # pin'3'    
                """
        for i in range(4,len(self.gridParameters['netInfo'][0])):
            pinCoord = self.gridParameters['netInfo'][0][str(i-3)]
            axes[pinCoord[2]].plot(pinCoord[0],pinCoord[1],'x')
        plt.show()
        return
    def pin_density_plot(self):
        pin_XList = []; pin_YList = []
        for i in self.gridParameters['netInfo']:
            for j in range(i['numPins']):
                pin_XList.append(i[str(j + 1)][0])
                pin_YList.append(i[str(j + 1)][1])
        plt.xlim([0, self.gridParameters['tileWidth'] * self.gridParameters['gridSize'][0]])
        plt.ylim([0, self.gridParameters['tileHeight'] * self.gridParameters['gridSize'][1]])
        plt.plot(pin_XList,pin_YList,'b.')
        plt.xlabel('X Lengths')
        plt.ylabel('Y Lengths')
        plt.show()
        return

    def bounding_length(self):
        j = 0
        halfPerimeterList = {}
        for net in self.gridParameters['netInfo']:
            netX = []; netY = []
            halfPerimeter = 0
            for i in range(net['numPins']):     #how many pins in one net
                netX.append(net[str(i+1)][0]) #              x      y   layer
                netY.append(net[str(i+1)][1]) #ex.pin['1']=['17', '62', '1'], 
            Xmin = min(netX); Xmax = max(netX)
            Ymin = min(netY); Ymax = max(netY)
            halfPerimeter = (Ymax - Ymin) + (Xmax - Xmin)   #*  shortest path
            halfPerimeterList[str(j)] = halfPerimeter
            j += 1
        return halfPerimeterList


def read(grfile):
    """
    input.gr
        grid 8 8 2\n
        vertical capacity 0 4\n
        horizontal capacity 4 0\n
        ...  \n
    return\n
        {0: ['grid', '8', '8', '2'], \n
        1: ['vertical', 'capacity', '0', '4'], \n
        2: ['horizontal', 'capacity', '4', '0'],\n
        ...
        }
    """
    file = open(grfile,'r')
    grid_info = {}
    i = 0
    for line in file:
        if not line.strip():
            continue
        else:
            grid_info[i]= line.split()
        i += 1
    file.close()
    return grid_info

# Parsing input data()
def gridParameters(grid_info):
    """
    {0: ['grid', '8', '8', '2'], \n
    1: ['vertical', 'capacity', '0', '4'], \n
    2: ['horizontal', 'capacity', '4', '0'],\n
3: ['minimum', 'width', '1', '1'], \n
4: ['minimum', 'spacing', '0', '0'], \n
5: ['via', 'spacing', '0', '0'], \n
6: ['0', '0', '10', '10'],  Origin  tileWidth tileHeight\n
7: ['num', 'net', '20'], numNet\n
'netInfo'[
    {8: ['A1', '01', '3', '1'],  #'netName''netID''numPins''minWidth'
          9: ['17', '62', '1'],     # pin'1'
          10: ['69', '30', '1'],    # pin'2'
          11: ['22', '45', '1'],    # pin'3'
    }
    {12: ['A2', '02', '3', '1'], 
        13: ['67', '39', '1'], 
        14: ['35', '11', '1'], 
        15: ['78', '22', '1'], 
    }
    {16: ['A3', '03', '4', '1'], 
        17: ['17', '54', '1'], 
        18: ['79', '68', '1'],
    }    
    ...
    ]
    reducedCapacitySpecify
        2 2 2   2 3 2   3
        2 5 2   2 6 2   3
        2 1 2   2 2 2   3
    }    
    """
    gridParameters = {}
    gridParameters['gridSize'] = [int(grid_info[0][1]),int(grid_info[0][2]),int(grid_info[0][3])]
    gridParameters['verticalCapacity'] = [float(grid_info[1][2]),float(grid_info[1][3])]
    gridParameters['horizontalCapacity'] = [float(grid_info[2][2]), float(grid_info[2][3])]
    gridParameters['minWidth'] = [float(grid_info[3][2]), float(grid_info[3][3])]
    gridParameters['minSpacing'] = [float(grid_info[4][2]), float(grid_info[4][3])]
    gridParameters['viaSpacing'] = [float(grid_info[5][2]), float(grid_info[5][3])]
    gridParameters['Origin'] = [float(grid_info[6][0]), float(grid_info[6][1])]
    gridParameters['tileWidth'] = float(grid_info[6][2]); gridParameters['tileHeight'] = float(grid_info[6][3])
    gridParameters['reducedCapacitySpecify'] = {}
    for lineNum in range(len(grid_info)):
        if 'num' in grid_info[lineNum]:
            gridParameters['numNet'] = int(grid_info[lineNum][2])   # --netNum 20
    lineEnumerator = 8
    netParametersStore = []
    for lineNum in range(7,len(grid_info)):
        if 'A' in grid_info[lineNum][0]:
            netParameters = {}
            netParameters['netName'] = grid_info[lineNum][0]          #A1
            netParameters['netID'] = int(grid_info[lineNum][1])       #01
            netParameters['numPins'] = int(grid_info[lineNum][2])     #3
            netParameters['minWidth'] = float(grid_info[lineNum][3])  #1
            pinNum = 1
            while ('A' not in grid_info[lineNum+pinNum][0]) and (len(grid_info[lineNum+pinNum])>1):
                netParameters[str(pinNum)] = [int(grid_info[lineNum+pinNum][0]),int(grid_info[lineNum+pinNum][1]),
                                             int(grid_info[lineNum+pinNum][2])]
                pinNum += 1
            gridParameters['reducedCapacity'] = grid_info[lineNum+pinNum]  #8+4 = 12 #A1's next = ['A2', '02', '3', '1'], 
            lineEnumerator = lineNum + pinNum + 1   #13
            netParametersStore.append(netParameters)
        if ('n' in grid_info[lineNum][0])and (grid_info[lineNum][0] != 'num'):
            netParameters = {}
            netParameters['netName'] = grid_info[lineNum][0]
            netParameters['netID'] = int(grid_info[lineNum][1])
            netParameters['numPins'] = int(grid_info[lineNum][2])
            netParameters['minWidth'] = float(grid_info[lineNum][3])
            pinNum = 1
            while ('n' not in grid_info[lineNum+pinNum][0]) and (len(grid_info[lineNum+pinNum])>1):
                netParameters[str(pinNum)] = [int(grid_info[lineNum+pinNum][0]),int(grid_info[lineNum+pinNum][1]),
                                             int(grid_info[lineNum+pinNum][2])]
                pinNum += 1
            gridParameters['reducedCapacity'] = grid_info[lineNum+pinNum]
            lineEnumerator = lineNum + pinNum + 1
            netParametersStore.append(netParameters)
        gridParameters['netInfo'] = netParametersStore
    # Parsing adjustments depicting reduced capacity (override layer specification)
    i = 1
    for lineNum in range(lineEnumerator, len(grid_info)):
        reducedEdge = [int(grid_info[lineNum][0]),int(grid_info[lineNum][1]),int(grid_info[lineNum][2]),
                       int(grid_info[lineNum][3]),int(grid_info[lineNum][4]),int(grid_info[lineNum][5]),
                       int(grid_info[lineNum][6])]
        gridParameters['reducedCapacitySpecify'][str(i)] = reducedEdge
            # grid_info[lineNum]
        # print(lineNum)
        i += 1
    '''
    the last 3 row
    ex. 
        2 2 2   2 3 2   3
        2 5 2   2 6 2   3
        2 1 2   2 2 2   3
    '''
    return gridParameters
#gridParameters(read("test_benchmark_1.gr"))
#
if __name__ == '__main__':
    # filename = 'adaptec1.capo70.2d.35.50.90.gr'
#     filename = 'sampleBenchmark'
    filename = 'small.gr'

    grid_info = read(filename)
    # print(grid_info)
    # print(gridParameters(grid_info)['netInfo'])
#
    for item in gridParameters(grid_info).items():
        print(item)
#     # for net in gridParameters(grid_info)['netInfo']:
#     #     print (net)

    # Testing visualization
    # VisualGraph(gridParameters(grid_info)).show_grid()
    # VisualGraph(gridParameters(grid_info)).pin_density_plot()

    halfParameterList = VisualGraph(gridParameters(grid_info)).bounding_length()
    print(halfParameterList)