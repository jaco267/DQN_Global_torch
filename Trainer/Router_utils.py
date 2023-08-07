from __future__ import print_function

import matplotlib;   matplotlib.use('TkAgg');

from Trainer import GridGraphV2 as graph
from Trainer import MST as tree

def gen_2pinListClear(gridParameters):
    twopinListComboCleared = []
    for i in range(len(gridParameters['netInfo'])):   # for net num == 20
        netNum = i
        netPinList = []; netPinCoord = []
        for j in range(0, gridParameters['netInfo'][netNum]['numPins']):
            pin = tuple([int((gridParameters['netInfo'][netNum][str(j+1)][0]-gridParameters['Origin'][0])/gridParameters['tileWidth']),
                                int((gridParameters['netInfo'][netNum][str(j+1)][1]-gridParameters['Origin'][1])/gridParameters['tileHeight']),
                                int(gridParameters['netInfo'][netNum][str(j+1)][2]),
                                int(gridParameters['netInfo'][netNum][str(j+1)][0]),
                                int(gridParameters['netInfo'][netNum][str(j+1)][1])])
            if pin[0:3] in netPinCoord:
                continue
            else:
                netPinList.append(pin)
                netPinCoord.append(pin[0:3])
        twoPinList = []
        for i in range(len(netPinList)-1):
            pinStart = netPinList[i]
            pinEnd = netPinList[i+1]
            twoPinList.append([pinStart,pinEnd])

        twoPinListVanilla = twoPinList

        # Insert Tree method to decompose two pin problems here
        twoPinList = tree.generateMST(twoPinList)
#        print('Two pin list after:', twoPinList, '\n')

        # Remove pin pairs that are in the same grid 
        nullPairList = []
        for i in range(len(twoPinListVanilla)):
            if twoPinListVanilla[i][0][:3] == twoPinListVanilla[i][1][:3]:
                nullPairList.append(twoPinListVanilla[i])
        for i in range(len(nullPairList)):
            twoPinListVanilla.reomove(nullPairList[i])

        # Remove pin pairs that are in the same grid 
        nullPairList = []
        for i in range(len(twoPinList)):
            if twoPinList[i][0][:3] == twoPinList[i][1][:3]:
                nullPairList.append(twoPinList[i])
        for i in range(len(nullPairList)):
            twoPinList.reomove(nullPairList[i])

        # Key: use original sequence of two pin pairs
        twopinListComboCleared.append(twoPinListVanilla)


        twoPinEachNetClear = []
        for i in twopinListComboCleared:
            num = 0
            for j in i:
                num = num + 1
            twoPinEachNetClear.append(num)
    return twoPinEachNetClear

def gen2pinListCombo(gridParameters,sortedHalfWireLength):
    twopinListCombo = [];  
    for i in range(len(gridParameters['netInfo'])):
        netNum = int(sortedHalfWireLength[i][0]) # i 
        # Sort the pins by a heuristic such as Min Spanning Tree or Rectilinear Steiner Tree
        netPinList = [];       netPinCoord = [];
        for j in range(0, gridParameters['netInfo'][netNum]['numPins']):
            pin = tuple([int((gridParameters['netInfo'][netNum][str(j+1)][0]-gridParameters['Origin'][0])/gridParameters['tileWidth']),
                                int((gridParameters['netInfo'][netNum][str(j+1)][1]-gridParameters['Origin'][1])/gridParameters['tileHeight']),
                                int(gridParameters['netInfo'][netNum][str(j+1)][2]),
                                int(gridParameters['netInfo'][netNum][str(j+1)][0]),
                                int(gridParameters['netInfo'][netNum][str(j+1)][1])])
            if pin[0:3] in netPinCoord:
                continue
            else:
                netPinList.append(pin)
                netPinCoord.append(pin[0:3])

        twoPinList = []
        for i in range(len(netPinList)-1):
            pinStart = netPinList[i]
            pinEnd = netPinList[i+1]
            twoPinList.append([pinStart,pinEnd])

        twoPinList = tree.generateMST(twoPinList)

        # Remove pin pairs that are in the same grid 
        nullPairList = []
        for i in range(len(twoPinList)):
            if twoPinList[i][0][:3] == twoPinList[i][1][:3]:
                nullPairList.append(twoPinList[i])
        for i in range(len(nullPairList)):
            twoPinList.reomove(nullPairList[i])

        # Key: Use MST sorted pin pair sequence under half wirelength sorted nets
        twopinListCombo.append(twoPinList)
    return twopinListCombo

def gen2pinlistNet(twopinListCombo):
    twopinlist_nonet = []
    for net in twopinListCombo:
        for pinpair in net:
            twopinlist_nonet.append(pinpair)
    return twopinlist_nonet

def gen_data(repeat_buffer,
             gridParameters,routeListMerged,graphcaseBurnIn):
    observationCombo = []; actionCombo = []; rewardCombo = []
    observation_nextCombo = []; is_terminalCombo = []
    for enumerator in range(repeat_buffer):
        for i in range(gridParameters['numNet']):
            goal = routeListMerged[i][-1]
            graphcaseBurnIn.goal_state = (goal[3],goal[4],goal[2],goal[0],goal[1])
            for j in range(len(routeListMerged[i])-1):
                position = routeListMerged[i][j]
                nextposition = routeListMerged[i][j+1]
                graphcaseBurnIn.current_state = (position[3],position[4],
                    position[2],position[0],position[1])
                # print(graphcaseBurnIn.state2obsv())
                observationCombo.append(graphcaseBurnIn.state2obsv())
                action = graph.get_action(position,nextposition)
                # print('action',action)
                actionCombo.append(action)

                graphcaseBurnIn.step(action)
                rewardCombo.append(graphcaseBurnIn.instantreward)
                # graphcaseBurnIn.current_state = (nextposition[3],nextposition[4],
                #     nextposition[2],nextposition[0],nextposition[1])
                observation_nextCombo.append(graphcaseBurnIn.state2obsv())
                is_terminalCombo.append(False)

            is_terminalCombo[-1] = True

    return observationCombo , actionCombo , rewardCombo,\
          observation_nextCombo , is_terminalCombo,


