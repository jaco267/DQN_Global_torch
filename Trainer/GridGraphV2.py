import matplotlib
matplotlib.use('TkAgg')
import numpy as np


from Trainer import Initializer as init

# Create grid graph based on parsed input info
'''  in test_benchmark_1.gr
    'gridSize' '8', '8', '2'], \n
    'verticalCapacity' '0', '4'], \n
   'horizontalCapacity' '4', '0'],\n
    'minWidth' '1', '1'], \n
    'minSpacing' '0', '0'], \n
    'viaSpacing' '0', '0'], \n
    Origin  ['0', '0',]   \n
    tileWidth tileHeight  []'10', '10'], \n
    'numNet'  '20'\n
        'netName''netID''numPins''minWidth'
        ['A1', '01', '3', '1'], 
            ['17', '62', '1'],     # pin'1'
            ['69', '30', '1'],    # pin'2'
            ['22', '45', '1'],    # pin'3'
        ['A2', '02', '3', '1'], 
            ['67', '39', '1'], 
            ['35', '11', '1'], 
            15: ['78', '22', '1'], 
        ['A3', '03', '4', '1'], 
            ['17', '54', '1'], 
            ['79', '68', '1'],
        ...
    reducedCapacitySpecify
        
        2 2 2   2 3 2   3
        2 5 2   2 6 2   3
        2 1 2   2 2 2   3
    }    
'''
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 
class GridGraph(object):
    def __init__(self, gridParameters,max_step=100,twopin_combo=None,net_pair=[1,2,3]):
        '''
        twopin_combo
        ex. A1
        28  62 1  #start point                         4-1 = 3 connect                   
        30  17 1                 ->           [(2, 6, 1, 28, 62), (3, 1, 1, 30, 17)]                                    
        78  75 1                              [(2, 6, 1, 28, 62), (7, 7, 1, 78, 75)]                
        11  77 1                              [(2, 6, 1, 28, 62), (1, 7, 1, 11, 77)]    
        #                                       grid lay 2.8 6.2            1.1 1.7               
        netpair       ex.A1 4pin-1 = 3connect
        [3, 2, 2, 1, 4, 2, 3, 4,... 3, 1, 4] 
        '''
        print("\ngrid graphV2")
        self.max_step = max_step;  #*** 100
        self.gridParameters = gridParameters
        self.twopin_combo = twopin_combo; 
        self.net_pair = net_pair;
        self.capacity = self.generate_capacity()
             
        self.goal_state = None;   self.init_state = None;     
        self.net_ind = 0;   
        self.pair_ind = 0
        self.current_state = None
        self.current_step = 0;
        self.route = []
        self.passby = np.zeros_like(self.capacity)
        
        self.obs_size = 12   #in state2obsv()
        self.action_size = 6 #
        return
    def generate_capacity(self):   #*done
        """
        # Input: VerticalCapacity, HorizontalCapacity, ReducedCapacity, MinWidth, MinSpacing
        # Update Input: Routed Nets Path
        # Capacity description direction:
        #!  [0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z]
        """
        capacity = np.zeros((self.gridParameters['gridSize'][0],self.gridParameters['gridSize'][1],
                         self.gridParameters['gridSize'][2],6))     #*  8,8,2,6     ,gridx,gridy,up_down, 6:+-x,+-y,+-z

        ## Apply initial condition to capacity
        # Calculate Available NumNet in each direction
        # Layer 0
        verticalNumNet = [self.gridParameters['verticalCapacity'][0]/       #0 /  1+0
                          (self.gridParameters['minWidth'][0]+self.gridParameters['minSpacing'][0]),
                          self.gridParameters['verticalCapacity'][1] /      #4 /  1+0
                          (self.gridParameters['minWidth'][1] + self.gridParameters['minSpacing'][1])]
        horizontalNumNet = [self.gridParameters['horizontalCapacity'][0]/   #4 /  1+0
                          (self.gridParameters['minWidth'][0]+self.gridParameters['minSpacing'][0]),
                          self.gridParameters['horizontalCapacity'][1] /    #0 /  1+0
                          (self.gridParameters['minWidth'][1] + self.gridParameters['minSpacing'][1])]
        #* verticalNumNet = [0,4]  horizontalNumNet = [4,0]
        # print(horizontalNumNet)
        # Apply available NumNet to grid capacity variables
        capacity[:,:,0,0] = capacity[:,:,0,1] = horizontalNumNet[0] #4  layer0 +-x cap = 4   layer0  horizontal is open
        capacity[:,:,1,0] = capacity[:,:,1,1] = horizontalNumNet[1] #0     layer1 +-x cap = 0   layer1  horizontal is blocked
        capacity[:,:,0,2] = capacity[:,:,0,3] = verticalNumNet[0]        #0     layer0 +-y cap = 0   layer0's vertical is blocked
        capacity[:,:,1,2] = capacity[:,:,1,3] = verticalNumNet[1]        #4     layer1 +-y cap = 4   layer1  vertical is open

        # Assume Via Ability to be very large
        capacity[:,:,0,4] = 10;   #+z   layer0 +-z cap = 10
        capacity[:,:,1,5] = 10    #-z

        # Apply Reduced Ability
        for i in range(int(self.gridParameters['reducedCapacity'][0])):
            """
            reducedCapacitySpecify
                x y lay  delta  
                2 2 2   2 3 2   3
                2 5 2   2 6 2   3
                2 1 2   2 2 2   3
            """
            # print('Apply reduced capacity operation')
            delta = [self.gridParameters['reducedCapacitySpecify'][str(i + 1)][0]-               #  2-2
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][3],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][1] -              #2-3
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][4],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][2] -              #2-2
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][5],
                     ]        #delta = [0,-1,0]
            if delta[0] != 0:
                # print(self.gridParameters['reducedCapacitySpecify'][str(i + 1)][0]-1)
                capacity[self.gridParameters['reducedCapacitySpecify'][str(i + 1)][0],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][1],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][2]-1,
                     int((delta[0]+1)/2)] = \
                capacity[self.gridParameters['reducedCapacitySpecify'][str(i + 1)][3],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][4],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][5]-1,
                     int((-delta[0]+1)/2)] = \
                    self.gridParameters['reducedCapacitySpecify'][str(i + 1)][6]
            elif delta[1] != 0:                                                            #   x y layer   direct(6)
                capacity[self.gridParameters['reducedCapacitySpecify'][str(i + 1)][0],     #  [2,2, 2-1, (2+ (-1+1)/2)] = 2,3,2-1, (2+(1+1)/2) = 
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][1],         #  cap[2,2,1, +y]       = cap[2,3,2, -y] = 3     #*  build a hole between x,y(2,2,layer:1) +y direction      
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][2]-1,int(2+(delta[1]+1)/2)] = \
                capacity[self.gridParameters['reducedCapacitySpecify'][str(i + 1)][3],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][4],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][5]-1,int(2+(-delta[1]+1)/2)] = \
                    self.gridParameters['reducedCapacitySpecify'][str(i + 1)][6]
            elif delta[2] != 0:
                capacity[self.gridParameters['reducedCapacitySpecify'][str(i + 1)][0],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][1],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][2]-1,int(4+(delta[2]+1)/2)] = \
                    self.gridParameters['reducedCapacitySpecify'][str(i + 1)][6]
                capacity[self.gridParameters['reducedCapacitySpecify'][str(i + 1)][3],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][4],
                     self.gridParameters['reducedCapacitySpecify'][str(i + 1)][5]-1,int(4+(-delta[2]+1)/2)] = \
                    self.gridParameters['reducedCapacitySpecify'][str(i + 1)][6]

        #* Remove edge capacity
        # [0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z]
                    # layer1  +z     layer0  -z
        capacity[:,:,1,4] = 0; capacity[:,:,0,5] = 0 # Z-direction edge capacity edge removal
        capacity[:,0,:,3] = 0; capacity[:,self.gridParameters['gridSize'][1]-1,:,2] = 0 # Y-direction edge capacity edge removal
        capacity[0,:,:,1] = 0; capacity[self.gridParameters['gridSize'][0]-1,:,:,0] = 0 # X-direction edge capacity edge removal
        return capacity

    def step(self, action):  # used for DRL
        state = self.current_state
        """
                x, y, lay
        state: (4, 5, 1or2, 47.0, 57.0) ..sss
        action 0 # [0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z]
        """
        reward = -1            #cap of   x        y        lay 0or1    action
        if action == 0 and self.capacity[state[0], state[1], state[2]-1, 0] > 0 :     #*   +x
            nextState = (state[0]+1, state[1], state[2], state[3]+self.gridParameters['tileWidth'], state[4])
            if self.passby[state[0], state[1], state[2]-1, 0] == 0:
                self.passby[state[0], state[1], state[2]-1, 0]+=1
                self.passby[state[0]+1, state[1], state[2]-1, 1]+=1
                self.capacity = updateCapacityRL(self.capacity, state, action)
            self.route.append((state[3], state[4], state[2], state[0], state[1]))
        elif action == 1 and self.capacity[state[0], state[1], state[2]-1, 1] > 0 :   #*   -x
            nextState = (state[0]-1, state[1], state[2], state[3]-self.gridParameters['tileWidth'], state[4])
            if self.passby[state[0], state[1], state[2]-1, 1] == 0:
                self.passby[state[0], state[1], state[2]-1, 1]+=1
                self.passby[state[0]-1, state[1], state[2]-1, 0]+=1
                self.capacity = updateCapacityRL(self.capacity, state, action)
            self.route.append((state[3], state[4], state[2], state[0], state[1]))
        elif action == 2 and self.capacity[state[0], state[1], state[2]-1, 2] > 0 :   #*   +y
            nextState = (state[0], state[1]+1, state[2], state[3], state[4]+self.gridParameters['tileHeight'])
            if self.passby[state[0], state[1], state[2]-1, 2] == 0:
                self.passby[state[0], state[1], state[2]-1, 2]+=1
                self.passby[state[0], state[1]+1, state[2]-1, 3]+=1
                self.capacity = updateCapacityRL(self.capacity, state, action)
            self.route.append((state[3], state[4], state[2], state[0], state[1]))
        elif action == 3 and self.capacity[state[0], state[1], state[2]-1, 3] > 0 :    #*   -y
            nextState = (state[0], state[1]-1, state[2], state[3], state[4]-self.gridParameters['tileHeight'])
            if self.passby[state[0], state[1], state[2]-1, 3] == 0:
                self.passby[state[0], state[1], state[2]-1, 3]+=1
                self.passby[state[0], state[1]-1, state[2]-1, 2]+=1
                self.capacity = updateCapacityRL(self.capacity, state, action)
            self.route.append((state[3], state[4], state[2], state[0], state[1]))
        elif action == 4 and self.capacity[state[0], state[1], state[2]-1, 4] > 0 :    #*   +z
            nextState = (state[0], state[1], state[2]+1, state[3], state[4])
            if self.passby[state[0], state[1], state[2]-1, 4] == 0:
                self.passby[state[0], state[1], state[2]-1, 4]+=1
                self.passby[state[0], state[1], state[2], 5]+=1
                self.capacity = updateCapacityRL(self.capacity, state, action)
            self.route.append((state[3], state[4], state[2], state[0], state[1]))
        elif action == 5 and self.capacity[state[0], state[1], state[2]-1, 5] > 0 :    #*   -z
            nextState = (state[0], state[1], state[2]-1, state[3], state[4])
            if self.passby[state[0], state[1], state[2]-1, 5] == 0:
                self.passby[state[0], state[1], state[2]-1, 5]+=1
                self.passby[state[0], state[1], state[2]-2, 4]+=1
                self.capacity = updateCapacityRL(self.capacity, state, action)
            self.route.append((state[3], state[4], state[2], state[0], state[1]))
        else:
            nextState = state

        self.current_state = nextState
        self.current_step += 1
        done = False
        if self.current_state[:3] == self.goal_state[:3]:
            # print(self.max_step,"?????")
            done = True
            reward = self.max_step #*100   
            #reward =  min(100.0 - self.current_step,50)   #1.5?
            # print('complete->  step: ',self.current_step)
            self.route.append((self.current_state[3], self.current_state[4], self.current_state[2],
                               self.current_state[0], self.current_state[1]))

        elif self.current_step >= self.max_step:
            # reward = 0   #-200
            done = True
            # print("fail to complet,  step==max_step : ", self.current_step)
            self.route.append((self.current_state[3], self.current_state[4], self.current_state[2],
                               self.current_state[0], self.current_state[1]))
        
        return nextState, reward, done, []

    def reset(self, pin):
        '''
         self.twopin_combo   all pin  
        '''
        self.init_state = self.twopin_combo[pin][0]
        self.goal_state = self.twopin_combo[pin][1]
        self.current_state = self.init_state
        self.current_step = 0
        self.pair_ind += 1
        if self.pair_ind>= self.net_pair[self.net_ind]:
            self.net_ind += 1
            self.pair_ind = 0
            self.passby = np.zeros_like(self.capacity)
        ### Change Made
        self.route = []
        return self.current_state

    def state2obsv(self):                 # x  y lay x_  y_
        state = np.array(self.current_state)  #(2, 6, 1, 28, 62)
        capacity = np.squeeze(self.capacity[int(state[0]), int(state[1]), int(state[2])-1, :])  
        obs = np.concatenate((state[:3], capacity, self.goal_state[:3]), axis=0).reshape(-1) 
        return obs.reshape(1,-1)
    def sample(self):
        return np.random.randint(6)
def updateCapacityRL(capacity,state,action):
	# capacity could go to negative

	# capacity[xgrid,ygrid,z={0,1},0~5]
	# state [xgrid,ygrid,zlayer={1,2},xlength,ylength]
	# action
	if action == 0:
		capacity[state[0],state[1],state[2]-1,0] -= 1
		capacity[state[0]+1,state[1],state[2]-1,1] -= 1
	elif action == 1:
		capacity[state[0],state[1],state[2]-1,1] -= 1
		capacity[state[0]-1,state[1],state[2]-1,0] -= 1
	elif action == 2:
		capacity[state[0],state[1],state[2]-1,2] -= 1
		capacity[state[0],state[1]+1,state[2]-1,3] -= 1
	elif action == 3:
		capacity[state[0],state[1],state[2]-1,3] -= 1
		capacity[state[0],state[1]-1,state[2]-1,2] -= 1
	elif action == 4:
		capacity[state[0],state[1],state[2]-1,4] -= 1
		capacity[state[0],state[1],state[2],5] -= 1
	elif action == 5:
		capacity[state[0],state[1],state[2]-1,5] -= 1
		capacity[state[0],state[1],state[2]-2,4] -= 1
	return capacity

def get_action(position, nextposition):
    # position example (20,10,2,2,1)
    diff = (nextposition[3] - position[3],nextposition[4] - position[4] ,nextposition[2] - position[2])
    action = 0
    if diff[0] == 1:
        action = 0
    elif diff[0] == -1:
        action = 1
    elif diff[1] == 1:
        action = 2
    elif diff[1] == -1:
        action = 3
    elif diff[2] == 1:
        action = 4
    elif diff[2] == -1:
        action = 5
    return action

def newFunction():
    pass
 
