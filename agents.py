import torch
import torch.nn as nn

from random import random, randint


FORCE_CPU = True    # ignore GPU if set to True
COMPILE = False     # compile option not tested yet!

DEBUG = False  # set to True to output debug info

# --------------- end of config -------------

if FORCE_CPU:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:",device)


class Agent:

    no_of_instances = 0

    def get_config_options():
        return []

    def __init__(self) -> None:
        self.total_payoff = 0
        self.total_no_of_games_played = 0
        self.adversary_first_move_cooperations = 0

        self.no_of_encounters = 0

        self.instance_no = Agent.no_of_instances
        Agent.no_of_instances += 1


    def reset(self):
        self.history = []
        self.payoff = 0
        self.no_of_games_played = 0

        self.no_of_encounters += 1


    def get_decision(self, learning_phase, maturity):  # abstract method
        return None
    

    def set_result(self, result, payoff, learn=True):

        if result[1] and len(self.history)==0:  #  True = the adversary cooperated
            self.adversary_first_move_cooperations += 1

        self.history.append(result)
        self.no_of_games_played += 1
        self.payoff += payoff

        self.total_no_of_games_played += 1
        self.total_payoff += payoff



    def get_average_payoff(self):
        return self.payoff / self.no_of_games_played
    

    def get_total_average_payoff(self):
        if self.total_no_of_games_played != 0:
            return self.total_payoff / self.total_no_of_games_played
        else:
            return 0
        

    def get_adversary_first_game_cooperation_ratio(self):
        if self.total_no_of_games_played != 0:
            return self.adversary_first_move_cooperations / self.no_of_encounters
        else:
            return 0
        

    def __str__(self):
        return self.__class__.__name__+" ("+str(self.instance_no)+")"




class DefectAgent(Agent):

    def get_decision(self, learning_phase, maturity):
        return False


class CooperateAgent(Agent):

    def get_decision(self, learning_phase, maturity):
        return True


class TitForTatAgent(Agent):

    def get_decision(self, learning_phase, maturity):
        if len(self.history)==0:
            return True  # always start with cooperation!
        else:
            return self.history[-1][1]  # repeat opponents last decision


# a variant of the tit for tat agent which sometimes tries to get out of endless defection battles
class ForgivingTitForTatAgent(Agent):

    FORGIVING_PROB = 0.2

    def get_decision(self, learning_phase, maturity):
        if len(self.history)==0:
            return True  # always start with cooperation!
        else:
            if self.history[-1][1]: 
                return True
            else:
                if random()<ForgivingTitForTatAgent.FORGIVING_PROB:
                    return True
                else:
                    return False



class GrimTriggerAgent(Agent):

    def get_decision(self, learning_phase, maturity):
        if len(self.history)==0:
            return True  # start with cooperation
        else:
            # if the opponent defected once -> always defect from then on
            cooperate = True
            for h in self.history:
                if h[1] == False:
                    cooperate = False
            return cooperate


class RandomAgent(Agent):
    def get_decision(self, learning_phase, maturity):
        if random()<0.5:
            return True
        else:
            return False



# This elegant helper function is borrowed from the fantastic book "Inside Deep Learning" from Edward Raff. Highly recommended!
class LastTimeStep(nn.Module):
    def __init__(self, rnn_layers=1, bidirectional = False):
        super(LastTimeStep,self).__init__()
        self.rnn_layers = rnn_layers
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

    def forward(self, input):
        rnn_output = input[0]
        last_step = input[1]
        if (type(last_step)==tuple):
            last_step = last_step[0]
        batch_size = last_step.shape[1]
        last_step = last_step.view(self.rnn_layers, self.num_directions, batch_size, -1)
        last_step = last_step[ self.rnn_layers - 1 ]
        last_step = last_step.permute(1, 0, 2)
        return last_step.reshape(batch_size, -1)



class RNNPredictorAgent(Agent):

    D = 2
    CLASSES = 2

    MAX_HISTORY = 50  # we consider only the last MAX_HISTORY moves when predicting / learning

    def get_config_options():
        return ['hidden_nodes','learning_rate','clipping','bidirectional','layers']

    def __init__(self, model_type = "RNN", layers = 2, clipping = False, bidirectional = True, hidden_nodes=32, learning_rate=0.003):
        super(RNNPredictorAgent, self).__init__()

        self.MODEL_TYPE = model_type
        self.RNN_LAYERS = layers
        self.HIDDEN_NODES = hidden_nodes
        self.GRADIENT_CLIPPING = clipping
        self.BIDIRECTIONAL = bidirectional
        self.LEARNING_RATE = learning_rate

        if self.MODEL_TYPE == 'LSTM':
            self.rnn_model = nn.Sequential(
                nn.LSTM(self.D, self.HIDDEN_NODES, num_layers=self.RNN_LAYERS, batch_first = True, bidirectional=self.BIDIRECTIONAL),
                LastTimeStep(rnn_layers=self.RNN_LAYERS, bidirectional=self.BIDIRECTIONAL),
                nn.Linear(self.HIDDEN_NODES*( 2 if self.BIDIRECTIONAL else 1 ), self.CLASSES)
            )
        else:
            self.rnn_model = nn.Sequential(
                nn.RNN(self.D, self.HIDDEN_NODES, num_layers=self.RNN_LAYERS, batch_first = True, bidirectional=self.BIDIRECTIONAL),
                LastTimeStep(rnn_layers=self.RNN_LAYERS, bidirectional=self.BIDIRECTIONAL),
                nn.Linear(self.HIDDEN_NODES*( 2 if self.BIDIRECTIONAL else 1 ), self.CLASSES)
            )

        if self.GRADIENT_CLIPPING:
            for p in self.rnn_model.parameters():
                p.register_hook(lambda grad: torch.clamp(grad,-5,5))

        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.rnn_model.parameters(), lr = self.LEARNING_RATE)

        if COMPILE:
            self.compiled_model = torch.compile(self.rnn_model)

        self.rnn_model.to(device)

        self.softmax = nn.Softmax(dim=1)

        self.no_of_predictions = 0
        self.no_of_correct_predictions = 0
        self.last_prediction = None


    def save_last_prediction(self,prediction):
        self.no_of_predictions += 1
        p_coop,p_defect = prediction
        if p_coop>=p_defect:
            self.last_prediction = True
        else:
            self.last_prediction = False


    def update_stats(self,result):
        if result[1] == self.last_prediction:
            self.no_of_correct_predictions += 1


    def get_prediction_accuracy(self):
        return self.no_of_correct_predictions / self.no_of_predictions
    

    def get_prediction(self, future_history=None):
        if future_history != None:
            history = future_history
        else:
            history = self.history

        # if the history is empty, make a random decision
        if len(history) == 0: # return a random prediction if no experience yet
            p_coop = random()
            p_defect = 1.0 - p_coop
            if future_history == None:
                self.save_last_prediction((p_coop,p_defect))
            return [p_coop,p_defect]
        else:
            # set inputs and labels
            h = [ [1 if r[0] else 0, 1 if r[1] else 0] for r in history[-RNNPredictorAgent.MAX_HISTORY:] ]
            inputs = torch.tensor(h, dtype = torch.float32)
            # add the batch dimension
            inputs = inputs.unsqueeze(0)
            # predict the other agents move
            self.rnn_model.eval() # set to model evaluation mode
            inputs= inputs.to(device)
            with torch.no_grad():
                if COMPILE:
                    y = self.compiled_model(inputs)
                else:
                    y = self.rnn_model(inputs)
            sm = self.softmax(y)  # we need probabilities
            result = (sm[0][0].item(),sm[0][1].item())  # batch 0, convert to float
            #if DEBUG:
            #    print('RNN inputs:',inputs,'output:',result)
            if future_history == None:
                self.save_last_prediction(result)
            return result


    def get_strategy(self,prediction):
        if prediction[0]>prediction[1]:  # probability of cooperation > probability of defection
            return True #cooperate
        else:
            return False #defect


    def get_decision(self, learning_phase, maturity):
            prediction = self.get_prediction()

            return self.get_strategy(prediction)


    def set_result(self, result, payoff, learn=True):

        self.update_stats(result)

        if learn and len(self.history)>0:
            # set inputs
            h = [ [1 if r[0] else 0, 1 if r[1] else 0] for r in self.history[-RNNPredictorAgent.MAX_HISTORY:] ]
            inputs = torch.tensor(h, dtype = torch.float32)
            # add the batch dimension
            inputs = inputs.unsqueeze(0)
            inputs = inputs.to(device)
            other_agent_cooperating = result[1]
            # set the labels
            if other_agent_cooperating:
                labels = torch.tensor([1,0], dtype = torch.float32) # that is what it SHOULD have predicted: 100% probability that the other will cooperate
            else:
                labels = torch.tensor([0,1], dtype = torch.float32)
            # add the batch dim to the labels
            labels = labels.unsqueeze(0)
            labels = labels.to(device)
            # train the rnn
            self.rnn_model.train()
            self.optimizer.zero_grad()
            if COMPILE:
                y_hat = self.compiled_model(inputs)
            else:
                y_hat = self.rnn_model(inputs)
            loss = self.loss_func(y_hat, labels)
            loss.backward()
            self.optimizer.step()

        # now store the result in the history etc. (by calling the superclass method)
        super(RNNPredictorAgent, self).set_result(result, payoff)



class LSTMPredictorAgent(RNNPredictorAgent):

    def __init__(self, layers = 2, clipping = False, bidirectional = True, hidden_nodes=32, learning_rate=0.003):
        super(LSTMPredictorAgent, self).__init__(model_type="LSTM", layers = layers, clipping = clipping, bidirectional = bidirectional, hidden_nodes=hidden_nodes, learning_rate=learning_rate)



# this is an optimistic RNNPredictorAgent (first move is always cooperative instead of random)
class OptimisticRNNPredictorAgent(RNNPredictorAgent):

    def get_strategy(self, prediction):
        if len(self.history) == 0:  # first move should be always cooperation
            return True # cooperate
        else:
            if prediction[0]>prediction[1]:  # probability of cooperation > probability of defection
                return True #cooperate
            else:
                return False #defect


# additional option for threshold for decision based on cooperation probability
class ThresholdedOptimisticRNNPredictorAgent(OptimisticRNNPredictorAgent):

    def get_config_options():
        return super(ThresholdedOptimisticRNNPredictorAgent,ThresholdedOptimisticRNNPredictorAgent).get_config_options() + ['threshold']
    

    def __init__(self, **kwargs):
        self.threshold = kwargs['threshold']
        kwargs.pop('threshold', None)
        super(ThresholdedOptimisticRNNPredictorAgent, self).__init__(**kwargs)


    def get_strategy(self, prediction):
        if len(self.history) == 0:  # first move should be always cooperation
            return True # cooperate
        else:
            if prediction[0]>(self.threshold + prediction[1]):  # probability of cooperation > probability of defection
                return True #cooperate
            else:
                return False #defect



class PredictionNode():

    def __init__(self, node_history, depth, predictor_function, max_depth=2, p_coop_start=None):
        self.node_history = node_history
        self.depth = depth
        self.predictor_function = predictor_function
        self.max_depth = max_depth
        self.p_coop_start = p_coop_start

        if DEBUG and self.depth == 0:
            print('---Created---')

        # predict opponent move given this history
        if len(self.node_history)==0:
            if p_coop_start == None:  # if p_coop for first move not known -> make a random choice
                if random()>=0.5:
                    self.p_coop = 0.51 # so we actually don't know
                else:
                    self.p_coop = 0.49 # so we actually don't know
            else:
                self.p_coop = p_coop_start
        else:
            p  = self.predictor_function(future_history = self.node_history)
            self.p_coop = p[0]

        self.p_defect = 1 - self.p_coop

        if self.p_coop>=0.5:
            self.cooperation_expected = True
            self.probab = self.p_coop
        else:
            self.cooperation_expected = False
            self.probab = self.p_defect


    def scalar_mult(a_list,scalar):
        return [ r * scalar for r in a_list ]
    

    def scalar_add(a_list,scalar):
        return [ (r + scalar) for r in a_list ]
    

    def avg(a_list,b_list):
        avg_list = []
        for i,a in enumerate(a_list):
            avg_list.append((a + b_list[i])/2)
        return avg_list


    def get_payout(me, other):
        if me and other:
            payout = -2
        if (not me) and (not other):
            payout = -5
        if me and (not other):
            payout = -10
        if (not me) and other:
            payout = 0
        return payout


    def get_result(self):

        p_coop = self.p_coop
        p_defect = self.p_defect
        cooperation_expected = self.cooperation_expected

        if DEBUG:
            print('Depth:',self.depth)
            print('History:',self.node_history)
            print('Prediction:',p_coop,p_defect,'->',cooperation_expected)

        # now we have two options to react (same goes for the opponent):
        future_history_cooperating_cooperating = self.node_history + [(True, True)]
        future_history_defecting_cooperating = self.node_history + [(False, True)]
        future_history_cooperating_defecting = self.node_history + [(True, False)]
        future_history_defecting_defecting = self.node_history + [(False, False)]

        # the payouts for this options
        payout_cooperating_cooperating = PredictionNode.get_payout(True,True)
        payout_defecting_cooperating = PredictionNode.get_payout(False,True)
        payout_cooperating_defecting = PredictionNode.get_payout(True,False)
        payout_defecting_defecting = PredictionNode.get_payout(False,False)

        if self.depth < self.max_depth:  # create subtrees as long as max_depth is not reached

            if cooperation_expected:  # only follow the more likely subtrees
                node_cooperating_cooperating = PredictionNode(future_history_cooperating_cooperating, self.depth+1, self.predictor_function, self.max_depth).get_result()
                node_defecting_cooperating = PredictionNode(future_history_defecting_cooperating, self.depth+1, self.predictor_function, self.max_depth).get_result()

                if p_defect>0.0:  # this should speed things up quite a bit!
                    node_cooperating_defecting = PredictionNode(future_history_cooperating_defecting, self.depth+1, self.predictor_function, self.max_depth).get_result()
                    node_defecting_defecting = PredictionNode(future_history_defecting_defecting, self.depth+1, self.predictor_function, self.max_depth).get_result()
                else:
                    node_cooperating_defecting = 0
                    node_defecting_defecting = 0
            else:
                node_cooperating_defecting = PredictionNode(future_history_cooperating_defecting, self.depth+1, self.predictor_function, self.max_depth).get_result()
                node_defecting_defecting = PredictionNode(future_history_defecting_defecting, self.depth+1, self.predictor_function, self.max_depth).get_result()

                if p_coop>0.0:  # we expect defecting
                    node_cooperating_cooperating = PredictionNode(future_history_cooperating_cooperating, self.depth+1, self.predictor_function, self.max_depth).get_result()
                    node_defecting_cooperating = PredictionNode(future_history_defecting_cooperating, self.depth+1, self.predictor_function, self.max_depth).get_result()
                else:
                    node_cooperating_cooperating = 0
                    node_defecting_cooperating = 0

            node_cooperating = p_coop*node_cooperating_cooperating + p_defect*node_cooperating_defecting
            node_defecting = p_coop*node_defecting_cooperating + p_defect*node_defecting_defecting

            payout_cooperating = p_coop*payout_cooperating_cooperating + p_defect*payout_cooperating_defecting
            payout_defecting = p_coop*payout_defecting_cooperating + p_defect*payout_defecting_defecting

            if self.depth == 0: # ROOT NODE
                result = (node_cooperating + payout_cooperating, node_defecting + payout_defecting)

                if DEBUG:
                    print('Return (Root):',result)

                return result
            else: # INTERMEDIATE NODE
                # the nodes between root node and terminal node

                result = max(node_cooperating + payout_cooperating, node_defecting + payout_defecting)

                if DEBUG:
                    print('Return (Intermediate):',result)
                return result
        else: # TERMINAL NODE
            # max_depth reached -> calculate payouts for terminal nodes
            """
            if cooperation_expected:
                result = 0 # max(payout_cooperating_cooperating, payout_defecting_cooperating)  # always 0
            else:
                result = -1 # max(payout_cooperating_defecting, payout_defecting_defecting) # always -5
            """
            result = self.p_defect*-5 #in full: result = self.p_coop*0 + self.p_defect*-5 (but first term is always zero!)

            if DEBUG:
                print('Return (Terminal):',result)
            return result



class LookAheadRNNPredictorAgent(RNNPredictorAgent):

    EXPLORATION_PROBABILITY = 0.3

    FIXED_LEARNING_PHASE_STRATEGY = False

    def explore_value(optimal_value, learning_phase):
        """
        # in this variant we inverte optimal_value sometimes
        if learning_phase and (random() < LookAheadRNNPredictorAgent.EXPLORATION_PROBABILITY):
            return (not optimal_value)
        else:
            return optimal_value

        # in this variant we make the agent more optimistic during learning
        if optimal_value:
            return optimal_value
        else:
            if learning_phase and (random() < LookAheadRNNPredictorAgent.EXPLORATION_PROBABILITY):
                return (not optimal_value)
            else:
                return optimal_value
        """
        return optimal_value



    def get_config_options():
        return super(LookAheadRNNPredictorAgent,LookAheadRNNPredictorAgent).get_config_options() + ['lookahead_depth']
    

    def __init__(self, **kwargs):
        if 'lookahead_depth' in kwargs:
            self.lookahead_depth = kwargs['lookahead_depth']
        else:
            self.lookahead_depth = 3
        kwargs.pop('lookahead_depth', None)
        super(LookAheadRNNPredictorAgent, self).__init__(**kwargs)


    def get_decision(self, learning_phase, maturity):

        prediction = self.get_prediction()  # get_prediction is also called to get some accuracy stats for the neural network ( -> needs to be called always)

        if LookAheadRNNPredictorAgent.FIXED_LEARNING_PHASE_STRATEGY and learning_phase:
            if len(self.history)==0: # choose a new learning strategy for each encounter
                self.learning_strategy = randint(1,3)

            if random() < LookAheadRNNPredictorAgent.EXPLORATION_PROBABILITY: # explore sometimes during training to get a rich training experience
                if random()<0.5:  # exploration mode -> decide randomly!
                    return True
                else:
                    return False
            else:  # use an established strategy
                if self.learning_strategy == 1:
                    # behave like the tit for tat agent
                    if len(self.history)==0:
                        return True  # always start with cooperation!
                    else:
                        return self.history[-1][1]  # repeat opponents last decision
                elif self.learning_strategy == 2:
                    # behave like the OptimisticRNNPredictorAgent !
                    if len(self.history) == 0:  # first move should be always cooperation
                        return True # cooperate
                    else:
                        if prediction[0]>prediction[1]:  # probability of cooperation > probability of defection
                            return True #cooperate
                        else:
                            return False #defect
                else:
                    # behave like the grimmtrigger agent
                    cooperate = True
                    for h in self.history:
                        if h[1] == False:
                            cooperate = False
                    return cooperate

        # create a binary tree of different game histories (starting from now)
        payouts = PredictionNode(self.history, 0, self.get_prediction,
            self.lookahead_depth,
            self.get_adversary_first_game_cooperation_ratio() ).get_result()

        if payouts[0] >= payouts[1]:  # cooperating brings a better expected future reward than defecting, cooperate also in case (payout_coop == payout_defect)!
            return LookAheadRNNPredictorAgent.explore_value(True, learning_phase)
        else:
            if learning_phase and (random() < LookAheadRNNPredictorAgent.EXPLORATION_PROBABILITY*(1-maturity)):
                return LookAheadRNNPredictorAgent.explore_value(True, learning_phase)
            else:
                return LookAheadRNNPredictorAgent.explore_value(False, learning_phase)



class SmartLearnLookAheadRNNPredictorAgent(LookAheadRNNPredictorAgent):

    def get_decision(self, learning_phase, maturity):
        if learning_phase:
            # behave randomly to learn as many different behaviours as possible
            if random()<0.5:
                return True
            else:
                return False
            """
            # use tit for tat in childhood!
            if len(self.history)==0:
                return True  # always start with cooperation!
            else:
                return self.history[-1][1]  # repeat opponents last decision
            """
        else:
            # create a binary tree of different game histories (starting from now)
            payouts = PredictionNode(self.history, 0, self.get_prediction, max_depth = self.lookahead_depth).get_result()

            #print(payouts)

            if payouts[0] >= payouts[1]:  # cooperating brings a better average reward than defecting
                return True
            else:
                return False
