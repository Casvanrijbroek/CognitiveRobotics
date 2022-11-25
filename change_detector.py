import nengo
import nengo.spa as spa
import nengo_gui

D = 128  # the dimensionality of the vectors
n = 200  # number of neurons in ensembles

model = spa.SPA()
with model:
    # Deals with the detection of a change
    model.input_ = nengo.Node(0)  # Input slider
    model.change_input = nengo.Ensemble(n_neurons=n, dimensions=1)  # Neuron representation of input
    nengo.Connection(model.input_, model.change_input)

    def detect_change(x):
        """Output the difference between inputs"""
        return x[0] - x[1]


    def threshold(x):
        """Fire if above (high) absolute threshold.

        I chose 0.8 to make sure the firing signal doesn't fire on small changes and to make it shorter
        """
        if abs(x) > 0.8:
            return 1
        else:
            return 0

    model.change_detector = nengo.Ensemble(n_neurons=n, dimensions=2)  # Neuron representation of difference between
    #  the value of the current input and of a short while ago
    model.change_threshold = nengo.Ensemble(n_neurons=n, dimensions=1)  # Fires if difference is high enough
    nengo.Connection(model.change_input, model.change_detector[0])
    nengo.Connection(model.change_input, model.change_detector[1],
                     synapse=0.05)  # Very small synapse to keep the signal short
    nengo.Connection(model.change_detector, model.change_threshold,
                     function=detect_change)

    model.change_output = nengo.Ensemble(n_neurons=n, dimensions=1)  # Output for change detector (redundant?)
    nengo.Connection(model.change_threshold, model.change_output, function=threshold)

    letterVocab = spa.Vocabulary(D)  # Letter vocab
    letterVocab.parse("A+B+C+D+E+F")

    changeVocab = spa.Vocabulary(D)  # Change vocab
    changeVocab.parse("CHANGE")

    model.changeState = spa.State(D, vocab=changeVocab)  # State for detecting change

    model.cleanup = spa.AssociativeMemory(input_vocab=letterVocab, wta_output=True)  # Memory cleanup for output letter
    model.output = spa.State(D, vocab=letterVocab)  # Output state

    # Determine the weight of the change and the letter for the output rule
    change_cost = 0.8
    letter_cost = 0.2

    # Change to the next state (relative to the current state in cleanup memory!) if a change is detected
    actions = spa.Actions(
        f"{change_cost}*dot(changeState, CHANGE) + {letter_cost}*dot(cleanup, F) --> output=A",
        f"{change_cost}*dot(changeState, CHANGE) + {letter_cost}*dot(cleanup, A) --> output=B",
        f"{change_cost}*dot(changeState, CHANGE) + {letter_cost}*dot(cleanup, B) --> output=C",
        f"{change_cost}*dot(changeState, CHANGE) + {letter_cost}*dot(cleanup, C) --> output=D",
        f"{change_cost}*dot(changeState, CHANGE) + {letter_cost}*dot(cleanup, D) --> output=E",
        f"{change_cost}*dot(changeState, CHANGE) + {letter_cost}*dot(cleanup, E) --> output=F",
        "0.5 --> output=0"
    )
    model.bg = spa.BasalGanglia(actions)
    model.thalamus = spa.Thalamus(model.bg)

    nengo.Connection(model.change_output, model.changeState.input,
                     transform=changeVocab["CHANGE"].v.reshape(D, 1))  # Output change as a semantic pointer
    nengo.Connection(model.output.output, model.cleanup.am.input)  # Forward the current state to clean up memory
    nengo.Connection(model.output.output, model.output.input)  # Recursive connection to stay in current state

    nengo_gui.GUI(__file__).start()
