# ************************************************ Importing Libraries *************************************************
from environment.environment import Environment
from agent.agent import Agent
from seed import setup

from datetime import datetime as dt
from termcolor import colored
import os

# *********************************************** Setting Hyper-Parameters *********************************************
window_size = 15  # The number of prices insight a single state
ds_path = "data/EURUSD_Candlestick_1_Hour_BID_01.01.2006-31.12.2016-FILTERED.csv"  # The location of our dataset
n_episodes = 80  # Number of episodes to train our agent
pip_pos = 4  # The digit position of the current trade exchange where calculate the pips
trans_cost = 0  # The cost to do a single transaction (expressed in pips)
batch_size = 30  # The number of tuple (state, action, reward, next_state) to save before replay
stop_loss_value = -50  # The maximum loss that we can handle (expressed in pips)
performance_file_path = "performance/train_performance.txt" # Path where to store the training performance log file
log = "performance/train_log.txt"  # Path where to store the training log file
models_path = "models/"  # Path where are stored the models
n_prev_iterations = len(next(os.walk(models_path))[2])  # Get the number of existent models in the models_path
setup(seed_value=7)
# ********************************* Creating the Agent Model and the Environment Model *********************************
env = Environment(ds_path=ds_path, window_size=window_size, pip_pos=pip_pos, stop_loss=stop_loss_value,
                  trans_cost=trans_cost)
actions = env.get_actions()  # Getting the available action of the environment
agent = Agent(env.get_state_size(), env.get_actions_n())

if os.path.exists(performance_file_path):  # Checking if there are previous training performances saved
    os.remove(performance_file_path)  # Deleting the old train performances
if os.path.exists(log):  # Checking if there are previous training performances saved
    os.remove(log)  # Deleting the old train performances

print(dt.now())
print("stop loss:", stop_loss_value)
print("pc: BH")
# ********************************************* Looping over all Episodes ***************-******************************
for ep in range(n_episodes - n_prev_iterations):
    time_start = dt.now()
    total_revenue = 0  # Counts the total reward for a single episode
    print("Iteration: " + str(ep+1) + "/" + str(n_episodes - n_prev_iterations))
    env.reset()  # Resetting the environment
    agent.reset()  # Resetting the agent mini-batch memory
    state, reward = env.step("Hold")  # Making a first neutral action for get the first state

    # ******************************************* Looping over all Instances *******************************************
    while not env.done:  # Loop until we finish all the instances
        action = agent.act(state)  # The agent choose an action based on the current state
        next_state, reward = env.step(actions[action])  # Getting the next state and reward based on the action choose
        '''with open(log, "a+") as file:
            file.write(str(actions[action]) + "\n")  # Saving the performance on a file
            if env.stop_loss_triggered:
                file.write("Stop Loss Triggered!" + "\n")  # Saving the stop loss taken on a file
            file.write(str(reward) + "\n")  # Saving the performance on a file'''
        '''print(colored("Observation:", 'blue'), state)
        print(colored("Action:", 'yellow'), actions[action])
        if env.stop_loss_triggered:  # Alert when we got a stop loss from the environment
            print(colored('Stop loss triggered!', 'red'))
        print(colored("Next Observation:", 'blue'), next_state)
        print(colored("Reward:", 'cyan'), reward)'''

        total_revenue += reward

        agent.memory.append((state, action, reward, next_state))  # Saving the experience
        state = next_state

        if len(agent.memory) > batch_size:  # Making an analysis based on our experience
            agent.exp_replay(batch_size)

    total_revenue += state[0][-1]  # Get the last profit if the order still alive and the instances are over
    agent.model.save("models/model_ep" + str(ep + n_prev_iterations))  # Saving the weight of the NN-Agent

    # ***************************** Showing and Saving the Results over a Single Episode *******************************
    #print("----------------------------------------------------------------------------------------------------------")
    if total_revenue > 0:
        print(colored("Total Profit: ", 'blue'), colored(str(round(total_revenue, 1)), 'cyan'), "pips")
    else:
        print(colored("Total Profit: ", 'blue'), colored(str(round(total_revenue, 1)), 'red'), "pips")
    with open(performance_file_path, "a+") as file:
        file.write(str(round(total_revenue, 1)) + "\n")  # Saving the performance on a file
    print("Loss: " + str(round((agent.loss / env.get_n_instances()), 1)))
    time_stop = dt.now()
    print(colored("Execution time for this episode:", 'yellow'),
          round((time_stop - time_start).total_seconds(), 0), "seconds")
    print("-----------------------------------------------------------------------------------------------------------")

# ************************ Showing the Performance over all Episodes and Saving them on a File *************************
print("\n*************************************************************************************************************")
print("Recap of the profits over episodes:")
with open(performance_file_path, "r") as file:
    for performance in file:
        print(performance.rstrip("\n"))
print("***************************************************************************************************************")
