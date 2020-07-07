# ************************************************ Importing Libraries *************************************************
from keras.models import load_model

from agent.agent import Agent
from environment.environment import Environment
from seed import setup

import os
from datetime import datetime as dt
from termcolor import colored

import gc

# *********************************************** Setting Hyper-Parameters *********************************************
ds_path = "data/EURUSD_Candlestick_1_Hour_BID_01.01.2017-31.12.2017-FILTERED.csv"  # The location of our dataset
pip_pos = 4  # The digit position of the current trade exchange where calculate the pips
trans_cost = 0  # The cost to do a single transaction (expressed in pips)
batch_size = 30  # The number of tuple (state, action, reward, next_state) to save before replay
stop_loss_value = -50  # The maximum loss that we can handle (expressed in pips)
performance_file_path = "performance/testing_performance.txt"
models_path = "models/"
n_models = len(next(os.walk(models_path))[2])  # Get the number of existent models in the models_path
log = "performance/test_log.txt"
setup(seed_value=7)
# ********************************* Creating the Agent Model and the Environment Model *********************************

print(dt.now())
print("stop loss:", stop_loss_value)
print("pc: BH")

def evaluate(model_name):
	time_start = dt.now()

	model = load_model(model_name)  # Load the NN-agent model
	state_size = model.layers[0].input.shape.as_list()[1]  # Load the state size from the model
	window_size = int(state_size/2)
	env = Environment(ds_path=ds_path, window_size=window_size, pip_pos=pip_pos, stop_loss=stop_loss_value,
					  trans_cost=trans_cost)
	actions = env.get_actions()  # Getting the available actions of the environment
	actions_size = env.get_actions_n()  # Getting the number of the actions available into the environment

	agent = Agent(state_size=state_size, action_size=actions_size, is_eval=True, model_name=model_name)

	state, reward = env.step("Hold")  # Making a first neutral action for get the first state
	total_revenue = 0

	while not env.done:  # Loop until we finish all the instances

		action = agent.act(state)  # The agent choose an action based on the current state
		next_state, reward = env.step(actions[action])  # Getting the next state and reward based on the action choose
		#with open(log, "a+") as file:
			#file.write(str(actions[action]) + "\n")  # Saving the performance on a file
			#if env.stop_loss_triggered:
				#file.write("Stop Loss Triggered!" + "\n")  # Saving the stop loss taken on a file
			#file.write(str(reward) + "\n")  # Saving the performance on a file
		'''print(colored("Observation:", 'blue'), state)
		print(colored("Action:", 'yellow'), actions[action])
		if env.stop_loss_triggered:  # Alert when we got a stop loss from the environment
			print(colored('Stop loss triggered!', 'red'))
		print(colored("Next Observation:", 'blue'), next_state)
		print(colored("Reward:", 'cyan'), reward)'''

		total_revenue += reward

		#agent.memory.append((state, action, reward, next_state))  # Saving the experience
		state = next_state

		#if len(agent.memory) > batch_size:  # Making an analysis based on our experience
		#	agent.exp_replay(batch_size)

	# ***************************** Showing and Saving the Results over a Single Episode *******************************
	#print("-----------------------------------------------------------------------------------------------------------")
	if total_revenue > 0:
		print(colored("Total Profit: ", 'blue'), colored(str(round(total_revenue, 1)), 'cyan'), "pips")
	else:
		print(colored("Total Profit: ", 'blue'), colored(str(round(total_revenue, 1)), 'red'), "pips")
	with open(performance_file_path, "a+") as file:
		file.write(str(round(total_revenue, 1)) + "\n")  # Saving the performance on a file
	time_stop = dt.now()
	print(colored("Execution time for this episode:", 'yellow'),
		  round((time_stop - time_start).total_seconds(), 0), "seconds")
	#print("-----------------------------------------------------------------------------------------------------------")


if os.path.exists(performance_file_path):  # Checking if there are previous testing performances saved
	os.remove(performance_file_path)  # Deleting the old train performances
if os.path.exists(log):  # Checking if there are previous training performances saved
	os.remove(log)  # Deleting the old train performances

for n_mod in range(n_models):
	print("-----------------------------------------------------------------------------------------------------------")
	print("Evaluating Model", n_mod)
	evaluate(models_path + "model_ep" + str(n_mod))
	gc.collect()

