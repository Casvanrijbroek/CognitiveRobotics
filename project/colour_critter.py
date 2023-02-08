import grid
import nengo
import nengo.spa as spa
import numpy as np 


#we can change the map here using # for walls and RGBMY for various colours
mymap="""
#########
#R     Y#
# ##### #
# ##### #
# ##### #
# ##### #
# ##### #
#M  G  B#
#########
"""
mymap="""
##########
#R      Y#
# ## ### #
# ## ### #
# ## ### #
# ## ### #
# ## ### #
#M   B  G#
##########
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
    N = 500
    D = 64
    
    #All input nodes should feed into one ensemble. Here is how to do this for
    #the radar, see if you can do it for the others
    walldist = nengo.Ensemble(n_neurons=N, dimensions=3, radius=4)
    nengo.Connection(proximity_sensors, walldist)

    #For now, all our agent does is wall avoidance. It uses values of the radar
    #to: a) turn away from walls on the sides and b) slow down in function of 
    #the distance to the wall ahead, reversing if it is really close
    def movement_func(x):
        turn = x[2] - x[0]
        spd = (x[1] - 0.5)/2
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
    cur_col_ens = nengo.Ensemble(n_neurons=N, dimensions=3, radius=1.5)
    nengo.Connection(current_color, cur_col_ens)
    
    next_col_ens = nengo.Ensemble(n_neurons=N, dimensions=3, radius=1.5)
    nengo.Connection(ahead_color, next_col_ens)
    
    rgb_vocab = spa.Vocabulary(D)
    rgb_vocab.parse("BLUE+GREEN+RED")
    col_vocab = spa.Vocabulary(D)
    col_vocab.parse("BLUE+GREEN+RED+MAGENTA+YELLOW")
    
    model.cur_color = spa.State(D, vocab=col_vocab)
        
    model.cur_red = spa.State(D, vocab=rgb_vocab)
    nengo.Connection(cur_col_ens[0], model.cur_red.input,
                     transform=rgb_vocab["RED"].v.reshape(D, 1))
    model.cur_green = spa.State(D, vocab=rgb_vocab)
    nengo.Connection(cur_col_ens[1], model.cur_green.input,
                     transform=rgb_vocab["GREEN"].v.reshape(D, 1))
    model.cur_blue = spa.State(D, vocab=rgb_vocab)
    nengo.Connection(cur_col_ens[2], model.cur_blue.input,
                     transform=rgb_vocab["BLUE"].v.reshape(D, 1))

    model.next_color = spa.State(D, vocab=col_vocab)

    model.next_red = spa.State(D, vocab=rgb_vocab)
    nengo.Connection(next_col_ens[0], model.next_red.input,
                     transform=rgb_vocab["RED"].v.reshape(D, 1))
    model.next_green = spa.State(D, vocab=rgb_vocab)
    nengo.Connection(next_col_ens[1], model.next_green.input,
                     transform=rgb_vocab["GREEN"].v.reshape(D, 1))
    model.next_blue = spa.State(D, vocab=rgb_vocab)
    nengo.Connection(next_col_ens[2], model.next_blue.input,
                     transform=rgb_vocab["BLUE"].v.reshape(D, 1))
        
    model.seen_red = spa.State(D, vocab=col_vocab, feedback=1)
    model.seen_blue = spa.State(D, vocab=col_vocab, feedback=1)
    model.seen_green = spa.State(D, vocab=col_vocab, feedback=1)
    model.seen_yellow = spa.State(D, vocab=col_vocab, feedback=1)
    model.seen_magenta = spa.State(D, vocab=col_vocab, feedback=1)
    
    col_sequence = ["MAGENTA", "BLUE", "YELLOW", "GREEN", "RED"]
    obj_w = 0.4
    col_w = 0.8
    mem_w = 2
    color_memory_actions = spa.Actions(
        f"({obj_w}                                                        + {col_w}) * dot(cur_clean_color, {col_sequence[0]}) --> seen_{col_sequence[0].lower()}={mem_w} * {col_sequence[0]}",
        f"{obj_w} * dot(seen_{col_sequence[0].lower()}, {col_sequence[0]}) + {col_w} * dot(cur_clean_color, {col_sequence[1]}) --> seen_{col_sequence[1].lower()}={mem_w} * {col_sequence[1]}",
        f"{obj_w} * dot(seen_{col_sequence[1].lower()}, {col_sequence[1]}) + {col_w} * dot(cur_clean_color, {col_sequence[2]}) --> seen_{col_sequence[2].lower()}={mem_w} * {col_sequence[2]}",
        f"{obj_w} * dot(seen_{col_sequence[2].lower()}, {col_sequence[2]}) + {col_w} * dot(cur_clean_color, {col_sequence[3]}) --> seen_{col_sequence[3].lower()}={mem_w} * {col_sequence[3]}",
        f"{obj_w} * dot(seen_{col_sequence[3].lower()}, {col_sequence[3]}) + {col_w} * dot(cur_clean_color, {col_sequence[4]}) --> seen_{col_sequence[4].lower()}={mem_w} * {col_sequence[4]}",
        "0.8 --> ",
    )
    
    cur_color_recognition_actions = spa.Actions(
        "dot(cur_red, RED) - 0.05*(dot(cur_green, GREEN) - dot(cur_blue, BLUE)) --> cur_color=RED",
        "dot(cur_blue, BLUE) - 0.05*(dot(cur_green, GREEN) - dot(cur_red, RED)) --> cur_color=BLUE",
        "dot(cur_green, GREEN) - 0.05*(dot(cur_red, RED) - dot(cur_blue, BLUE)) --> cur_color=GREEN",
        "0.95*(dot(cur_red, RED) + dot(cur_green, GREEN)) - dot(cur_blue, BLUE) --> cur_color=YELLOW",
        "0.95*(dot(cur_red, RED) + dot(cur_blue, BLUE)) - dot(cur_green, GREEN) --> cur_color=MAGENTA",
        "0.8 --> cur_color=0",
    )

    next_color_recognition_actions = spa.Actions(
        "dot(next_red, RED) - 0.05*(dot(next_green, GREEN) - dot(next_blue, BLUE)) --> next_color=RED",
        "dot(next_blue, BLUE) - 0.05*(dot(next_green, GREEN) - dot(next_red, RED)) --> next_color=BLUE",
        "dot(next_green, GREEN) - 0.05*(dot(next_red, RED) - dot(next_blue, BLUE)) --> next_color=GREEN",
        "0.95*(dot(next_red, RED) + dot(next_green, GREEN)) - dot(next_blue, BLUE) --> next_color=YELLOW",
        "0.95*(dot(next_red, RED) + dot(next_blue, BLUE)) - dot(next_green, GREEN) --> next_color=MAGENTA",
        "0.8 --> next_color=0",
    )
    
    model.cur_clean_color = spa.AssociativeMemory(input_vocab=col_vocab,
                                                  wta_output=True)
    # Elongate the color signal with a weak feedback connection
    nengo.Connection(model.cur_clean_color.am.output, model.cur_clean_color.am.input, transform=0.5)
    
    model.next_clean_color = spa.AssociativeMemory(input_vocab=col_vocab,
                                                   wta_output=True)
    nengo.Connection(model.next_clean_color.am.output, model.next_clean_color.am.input, transform=0.5)
    
    model.cur_col_reg_bg = spa.BasalGanglia(cur_color_recognition_actions)
    model.cur_col_reg_thalamus = spa.Thalamus(model.cur_col_reg_bg)

    model.next_col_reg_bg = spa.BasalGanglia(next_color_recognition_actions)
    model.next_col_reg_thalamus = spa.Thalamus(model.next_col_reg_bg)
    
    model.col_mem_bg = spa.BasalGanglia(color_memory_actions)
    model.col_mem_thalamus = spa.Thalamus(model.col_mem_bg)

    cleanup_actions = spa.Actions(
        "cur_clean_color = cur_color",
        "next_clean_color = next_color",
    )
    
    model.cortical = spa.Cortical(cleanup_actions)


 