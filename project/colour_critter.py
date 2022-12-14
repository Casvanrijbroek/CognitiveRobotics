import grid
import nengo
import nengo.spa as spa
import numpy as np 


#we can change the map here using # for walls and RGBMY for various colours
mymap="""
#######
#  M  #
# # # #
# #B# #
#G Y R#
#######
"""
mymap2="""
#########
#G     Y#
# ##### #
# ##### #
# ##### #
# ##### #
# ##### #
#M  R  B#
#########
"""


#### Preliminaries - this sets up the agent and the environment ################ 
class Cell(grid.Cell):

    def color(self):
        if self.wall:
            return 'black'
        elif self.cellcolor == 1:
            return 'green'
        elif self.cellcolor == 2:
            return 'red'
        elif self.cellcolor == 3:
            return 'blue'
        elif self.cellcolor == 4:
            return 'magenta'
        elif self.cellcolor == 5:
            return 'yellow'
             
        return None

    def load(self, char):
        self.cellcolor = 0
        if char == '#':
            self.wall = True
            
        if char == 'G':
            self.cellcolor = 1
        elif char == 'R':
            self.cellcolor = 2
        elif char == 'B':
            self.cellcolor = 3
        elif char == 'M':
            self.cellcolor = 4
        elif char == 'Y':
            self.cellcolor = 5
            
            
world = grid.World(Cell, map=mymap, directions=int(4))

body = grid.ContinuousAgent()
world.add(body, x=1, y=2, dir=2)

#this defines the RGB values of the colours. We use this to translate the "letter" in 
#the map to an actual colour. Note that we could make some or all channels noisy if we
#wanted to
col_values = {
    0: [0.9, 0.9, 0.9], # White
    1: [0.2, 0.8, 0.2], # Green
    2: [0.8, 0.2, 0.2], # Red
    3: [0.2, 0.2, 0.8], # Blue
    4: [0.8, 0.2, 0.8], # Magenta
    5: [0.8, 0.8, 0.2], # Yellow
}

noise_val = 0.01 # how much noise there will be in the colour info

#You do not have to use spa.SPA; you can also do this entirely with nengo.Network()
model = spa.SPA()
with model:
    
    # create a node to connect to the world we have created (so we can see it)
    env = grid.GridNode(world, dt=0.005)

    ### Input and output nodes - how the agent sees and acts in the world ######

    #--------------------------------------------------------------------------#
    # This is the output node of the model and its corresponding function.     #
    # It has two values that define the speed and the rotation of the agent    #
    #--------------------------------------------------------------------------#
    def move(t, x):
        speed, rotation = x
        dt = 0.001
        max_speed = 20.0
        max_rotate = 10.0
        body.turn(rotation * dt * max_rotate)
        body.go_forward(speed * dt * max_speed)
        
    movement = nengo.Node(move, size_in=2)
    
    #--------------------------------------------------------------------------#
    # First input node and its function: 3 proximity sensors to detect walls   #
    # up to some maximum distance ahead                                        #
    #--------------------------------------------------------------------------#
    def detect(t):
        angles = (np.linspace(-0.5, 0.5, 3) + body.dir) % world.directions
        return [body.detect(d, max_distance=4)[0] for d in angles]
    proximity_sensors = nengo.Node(detect)

    #--------------------------------------------------------------------------#
    # Second input node and its function: the colour of the current cell of    #
    # agent                                                                    #
    #--------------------------------------------------------------------------#
    def cell2rgb(t):
        
        c = col_values.get(body.cell.cellcolor)
        noise = np.random.normal(0, noise_val,3)
        c = np.clip(c + noise, 0, 1)
        
        return c
        
    current_color = nengo.Node(cell2rgb)
     
    #--------------------------------------------------------------------------#
    # Final input node and its function: the colour of the next non-whilte     #
    # cell (if any) ahead of the agent. We cannot see through walls.           #
    #--------------------------------------------------------------------------#
    def look_ahead(t):
        
        done = False
        
        cell = body.cell.neighbour[int(body.dir)]
        if cell.cellcolor > 0:
            done = True 
            
        while cell.neighbour[int(body.dir)].wall == False and not done:
            cell = cell.neighbour[int(body.dir)]
            
            if cell.cellcolor > 0:
                done = True
        
        c = col_values.get(cell.cellcolor)
        noise = np.random.normal(0, noise_val,3)
        c = np.clip(c + noise, 0, 1)
        
        return c
        
    ahead_color = nengo.Node(look_ahead)    
    
    ### Agent functionality - your code adds to this section ###################
    
    #All input nodes should feed into one ensemble. Here is how to do this for
    #the radar, see if you can do it for the others
    walldist = nengo.Ensemble(n_neurons=500, dimensions=3, radius=4)
    nengo.Connection(proximity_sensors, walldist)

    #For now, all our agent does is wall avoidance. It uses values of the radar
    #to: a) turn away from walls on the sides and b) slow down in function of 
    #the distance to the wall ahead, reversing if it is really close
    def movement_func(x):
        turn = x[2] - x[0]
        spd = x[1] - 0.5
        return spd, turn
    
    #the movement function is only driven by information from the radar, so we
    #can connect the radar ensemble to the output node with this function 
    #directly. In the assignment, you will need intermediate steps
    nengo.Connection(walldist, movement, function=movement_func)
    
    # Try to create an extra identity connection (this greatly changes the behaviour)
    # Potentially troublesome for basing my decisions on multiple factors later
    # I should find a way to stabilize this
    #choose_movement = nengo.Ensemble(n_neurons=500, dimensions=3, radius=4)
    #nengo.Connection(walldist, choose_movement)
    #nengo.Connection(choose_movement, movement, function=movement_func)
    
    # Simple ensemble to represent the observed color
    col_ens = nengo.Ensemble(n_neurons=500, dimensions=3, radius=1.5)
    nengo.Connection(current_color, col_ens)
    
    D = 64
    
    rgb_vocab = spa.Vocabulary(D)
    rgb_vocab.parse("BLUE+GREEN+RED")
    col_vocab = spa.Vocabulary(D)
    col_vocab.parse("BLUE+GREEN+RED+MAGENTA+YELLOW+WHITE")
    
    model.color = spa.State(D, vocab=col_vocab)
        
    model.spa_red = spa.State(D, vocab=rgb_vocab)
    nengo.Connection(col_ens[0], model.spa_red.input, 
        transform=rgb_vocab["RED"].v.reshape(D, 1))
    model.spa_green = spa.State(D, vocab=rgb_vocab)
    nengo.Connection(col_ens[1], model.spa_green.input, 
        transform=rgb_vocab["GREEN"].v.reshape(D, 1))
    model.spa_blue = spa.State(D, vocab=rgb_vocab)
    nengo.Connection(col_ens[2], model.spa_blue.input, 
        transform=rgb_vocab["BLUE"].v.reshape(D, 1))
    
    actions = spa.Actions(
        "dot(spa_red, RED) - 0.05*(dot(spa_green, GREEN) - dot(spa_blue, BLUE)) --> color=RED",
        "dot(spa_blue, BLUE) - 0.05*(dot(spa_green, GREEN) - dot(spa_red, RED)) --> color=BLUE",
        "dot(spa_green, GREEN) - 0.15*(dot(spa_red, RED) - dot(spa_blue, BLUE)) --> color=GREEN",
        "0.95*(dot(spa_red, RED) + dot(spa_green, GREEN)) - dot(spa_blue, BLUE) --> color=YELLOW",
        "0.95*(dot(spa_red, RED) + dot(spa_blue, BLUE)) - dot(spa_green, GREEN) --> color=MAGENTA",
        "0.8 --> color=0",
    )
    
    model.cleanup = spa.AssociativeMemory(input_vocab=col_vocab, wta_output=True)
    nengo.Connection(model.color.output, model.cleanup.am.input)
    #nengo.Connection(model.color.output, model.color.input)
    
    model.memory = spa.State(D)
    nengo.Connection(model.cleanup.am.output, model.memory.input)
    nengo.Connection(model.memory.output, model.memory.input)
    
    model.bg = spa.BasalGanglia(actions)
    model.thalamus = spa.Thalamus(model.bg)





 