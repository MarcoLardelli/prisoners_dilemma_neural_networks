
from random import choice, choices, uniform, randint, seed
import time
import json
from tabulate import tabulate

import pandas as pd

from agents import DefectAgent,CooperateAgent,TitForTatAgent,ForgivingTitForTatAgent,GrimTriggerAgent,RandomAgent \
        ,RNNPredictorAgent,LSTMPredictorAgent,OptimisticRNNPredictorAgent,ThresholdedOptimisticRNNPredictorAgent \
        ,LookAheadRNNPredictorAgent,SmartLearnLookAheadRNNPredictorAgent

# --- configuration ---

NO_OF_AGENTS = 200  # low: better training for each agent, but worse stats (not so many agents)

NO_OF_TRAINING_EPISODES_FROM = 10000
NO_OF_TRAINING_EPISODES_TO = 10000

NO_OF_EVALUATION_EPISODES = 2000  # give the agents some time to learn before storing results for stats

STEPS_PER_EPISODE = 15

NO_OF_RUNS = 1

DEBUG = False


# -------- end of configuration ----------


def get_short_class_name(agent_class):
    return str(type(agent_class)).split('.')[1][0:-2]

def get_agent_class_index(agent):
    for i in range(len(agent_distribution)):
        if type(agent)==agent_distribution[i][0]:
            return i
    return None


def check_agent_distribution_validity(agent_distribution):
    # check validity of agent configuration
    total = 0
    for agent in agent_distribution:
        total += agent[1]
    if abs(1.0-total)>0.001:
        return False
    else:
        return True


def create_matrix_df(games_results):
    pandas_matrix = []
    i = 0
    for row in games_results:
        new_row = []
        for elem in row:
            if len(elem)>0:
                avg = sum(elem)/len(elem)
                new_row.append(avg)
            else:
                new_row.append(None)
        new_row = [get_short_class_name(agent_distribution[i][0]())] + new_row
        pandas_matrix.append(new_row)
        i += 1

    df = pd.DataFrame(pandas_matrix, columns = ['Agent']+[get_short_class_name(a[0]()) for a in agent_distribution])
    pd.options.display.max_rows = 12

    # add table headers in first row for output with tabulate
    tabulate_matrix = [ ['Agent Class']+[get_short_class_name(agent[0]()) for agent in agent_distribution] ] + pandas_matrix

    return df, tabulate_matrix

def run_games(agent_distribution, hyperparameters, no_of_training_episodes,run_no):
    global DEBUG
    if not check_agent_distribution_validity(agent_distribution):
        print("Invalid agent distribution")
        exit()

    # create the list of agent objects (according to agent configuration)
    agents = []
    for i in range(NO_OF_AGENTS):
        # find type of agent
        ag = [ a[0] for a in agent_distribution ]
        weights = [ a[1] for a in agent_distribution ]
        selected_agent = choices(ag,weights)[0]

        # get parameter list of agent class
        required_params = selected_agent.get_config_options()
        params = {}
        for p in required_params:
            params[p] = hyperparameters[p]
        # create agent
        agents.append(selected_agent(**params))

    #play the games

    # initialize results matrix
    games_results = []
    for agent in agent_distribution:
        games_results.append([[] for a in agent_distribution]) # do NOT use (no_of_columns * []) here (will all point to the same object!)!!!

    games_results_learn = []
    for agent in agent_distribution:
        games_results_learn.append([[] for a in agent_distribution]) # do NOT use (no_of_columns * []) here (will all point to the same object!)!!!

    # Start timer
    start_time = time.time()

    with open('debug_messages.txt', 'w') as f:
        for i in range(no_of_training_episodes + NO_OF_EVALUATION_EPISODES):
            learn = (i < no_of_training_episodes)
            maturity = i / no_of_training_episodes

            if (i % 1000) == 0:
                print("Episode:",i)
                if i>0:
                    if learn:
                        df,tabulate_matrix = create_matrix_df(games_results_learn)
                    else:
                        df,tabulate_matrix = create_matrix_df(games_results)
                    #print(df) # not good if too many columns
                    print('Preliminary results (training...):')
                    if learn:
                        print('(LEARNING PHASE)')
                    print(tabulate(tabulate_matrix, headers='firstrow'))
                    print('----------------------')

            # randomly select two (different!) agents
            while True:
                agent1 = choice(agents)
                agent2 = choice(agents)
                if agent1 != agent2:
                    break

            # let the two agents play STEPS_PER_EPISODE steps against each other
            agent1.reset()
            agent2.reset()

            #if not learn:
            #    DEBUG = True

            # not sure if this is really needed!
            if learn:
                # make sure NN is trained on future histories too
                steps_per_episode = STEPS_PER_EPISODE + (hyperparameters['lookahead_depth']+1)
            else:
                steps_per_episode = STEPS_PER_EPISODE

            for j in range(steps_per_episode):
                # get the decisions from both agents
                cooperating1 = agent1.get_decision(learn, maturity)
                cooperating2 = agent2.get_decision(learn, maturity)

                # some debugging infos
                if DEBUG or (not learn):

                    if agent1.__class__.__name__ == 'LookAheadRNNPredictorAgent':
                        print(str(agent1),' vs ',str(agent2), cooperating1, cooperating2, agent1.get_prediction_accuracy(),file=f)
                    if agent2.__class__.__name__ == 'LookAheadRNNPredictorAgent':
                        print(str(agent2),' vs ',str(agent1), cooperating2, cooperating1, agent2.get_prediction_accuracy(),file=f)
                    """
                    if agent1.__class__.__name__ == 'OptimisticRNNPredictorAgent':
                        print(agent2.__class__.__name__, cooperating1, cooperating2, agent1.get_prediction_accuracy())
                    if agent2.__class__.__name__ == 'OptimisticRNNPredictorAgent':
                        print(agent1.__class__.__name__, cooperating2, cooperating1, agent2.get_prediction_accuracy())
                    """

                # communicate the results back to the two agents
                if cooperating1 and cooperating2:
                    agent1.set_result((True, True),-2, learn)
                    agent2.set_result((True, True),-2, learn)
                elif (not cooperating1) and (not cooperating2):
                    agent1.set_result((False, False),-5, learn)
                    agent2.set_result((False, False),-5, learn)
                elif cooperating1 and (not cooperating2):
                    agent1.set_result((True, False),-10, learn)
                    agent2.set_result((False, True),0, learn)
                else: # (not cooperating1) and cooperating2
                    agent1.set_result((False, True),0, learn)
                    agent2.set_result((True, False),-10, learn)


            m = get_agent_class_index(agent1)
            n = get_agent_class_index(agent2)
            if learn:
                games_results_learn[m][n].append(agent1.get_average_payoff())
                games_results_learn[n][m].append(agent2.get_average_payoff())
            else:
                #store results in games_results2 matrix too
                games_results[m][n].append(agent1.get_average_payoff())
                games_results[n][m].append(agent2.get_average_payoff())


    end_time = time.time()

    elapsed_time = end_time - start_time

    # calculate average performance and average number of encounters of various agents
    performances = {}
    no_of_encounters = {}
    accuracies = {}
    for agent in agents:
        average_payoff = agent.get_total_average_payoff()
        agent_name = get_short_class_name(agent)
        if agent_name in performances:
            performances[agent_name].append(average_payoff)
            no_of_encounters[agent_name].append(agent.no_of_encounters)
        else:
            performances[agent_name] = [average_payoff]
            no_of_encounters[agent_name] = [agent.no_of_encounters]
        if agent.__class__.__name__ == 'RNNPredictorAgent' or issubclass(agent.__class__, RNNPredictorAgent):
            acc = agent.get_prediction_accuracy()
            if agent_name in accuracies:
                accuracies[agent_name].append(acc)
            else:
                accuracies[agent_name] = [acc]


    average_performances = {}
    average_no_of_encounters = {}
    average_accuracies = {}
    for p in performances:
        average_performances[p] = sum(performances[p])/len(performances[p])
    for p in no_of_encounters:
        average_no_of_encounters[p] = sum(no_of_encounters[p])/len(no_of_encounters[p])
    for p in accuracies:
            average_accuracies[p] = sum(accuracies[p])/len(accuracies[p])

    return elapsed_time, average_performances, games_results, average_no_of_encounters, accuracies








agent_distribution = [
    (DefectAgent,0.15),
    (CooperateAgent,0.15),
    (TitForTatAgent,0.15),
    (ForgivingTitForTatAgent,0.4),
    #(GrimTriggerAgent,0.1),
    (RandomAgent,0.05),  # this should also help as a "regularizer" (introduce some noise into the data)
    #(RNNPredictorAgent,0.1),  # the optimistic version is clearly better
    #(LSTMPredictorAgent,0.075),
    (OptimisticRNNPredictorAgent,0.05),
    (LookAheadRNNPredictorAgent,0.05),
    #(SmartLearnLookAheadRNNPredictorAgent,0.1)
    #(ThresholdedOptimisticRNNPredictorAgent,0.05)
]



seed(0)
run_results = []
for run_no in range(NO_OF_RUNS):

    hyperparameters = {
        'layers': choice([1,1]),  # smaller values seem to be better
        'hidden_nodes': choice([16,16]),  # larger values seem to be better
        'clipping': choice([False, False]),     # no difference, so save the compute
        'bidirectional': choice([False, False]),  # this does not seem to help
        'learning_rate': uniform(0.01,0.01),  # we can make this a bit larger
        'threshold': uniform(0.0,0.0),
        'lookahead_depth': choice([4,4])
    }

    no_of_training_episodes = randint(NO_OF_TRAINING_EPISODES_FROM,NO_OF_TRAINING_EPISODES_TO)  # more!

    print("Run no:",run_no,"No of training episodes:",no_of_training_episodes)
    print("Hyperparameters:",hyperparameters)
    elapsed_time, average_performances, games_results, average_no_of_encounters, accuracies = run_games(agent_distribution, hyperparameters, no_of_training_episodes, run_no)

    results = (no_of_training_episodes, hyperparameters, elapsed_time, average_performances, average_no_of_encounters, accuracies)

    # store data in list
    run_results.append(results)

    # save data to files (one for each run) for later analysis
    with open('run_'+str(run_no)+'_data.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # print results matrix
    print("elapsed time", elapsed_time)
    print("Average payouts (training AND evaluation!)")
    for p in average_performances:
        print(p, average_performances[p])

    df,tabulate_matrix = create_matrix_df(games_results)
    #print(df) # not good if too many columns
    print(' ')
    print(tabulate(tabulate_matrix, headers='firstrow'))
