from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque


class Agent:
	def __init__(self, state_size, action_size, is_eval=False, model_name=""):
		"""
		:param state_size: specific the size of a single state
		:param action_size: specific the number of all the possible actions
		:param is_eval: specific if the agent must act and learn (for training) or just act (for testing)
		:param model_name: specific the model path to load if we wanna just test our agent
		"""
		self.state_size = state_size  # Specific the size of observation coming from the environment
		self.action_size = action_size  # Specific the number of the available actions (buy_long, buy_short, hold, close)
		self.memory = deque(maxlen=1000)  # Make a fast list push-pop
		self.model_name = model_name  # Used when we want to use a specific model
		self.is_eval = is_eval  # If True the agent don't use random actions

		self.gamma = 0.98  # Indicate how much important are the future rewards
		self.epsilon = 1.0  # Indicates the probability to do a random choice instead a smart choice
		self.epsilon_min = 0.01  # Indicates the minimum probability to do a random choice
		self.epsilon_decay = 0.95  # The random chance decay over an entire replay of experience
		self.learning_rate = 0.0003  # Learning rate of the neural network
		self.firstIter = True  # Monitor if we are in our first action

		# Load previous models if exists (in case of test or recover training) or just make a new model
		self.model = load_model(model_name) if not model_name == "" else self._model()
		self.loss = 0

	def _model(self):
		"""
		This function return NN model with "state_size" input and "action_size" output

		:return: a compiled model
		"""
		model = Sequential()
		model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
		model.add(Dense(units=32, activation="relu"))
		model.add(Dense(units=16, activation="relu"))
		model.add(Dense(units=8, activation="relu"))
		model.add(Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

		return model

	def show_model_summary(self):
		"""
		This function show a NN model summary, visualizing the structure of the net

		:return: a summary of the compiled model
		"""
		return self.model.summary()

	'''def act(self, state):
		"""
		This function return NN model with "state_size" input and "action_size" output

		:param state: a tensor that represent a single observation of the environment

		:return: the action (prediction) with the best q-value  (the best action for this observation)
		"""
		rand_val = np.random.rand()
		if not self.is_eval and rand_val <= self.epsilon:  # Do a random action only in train phase
			return random.randrange(self.action_size)

		if self.firstIter:  # If this is the first iteration, just do a "hold" action
			self.firstIter = False
			return 2  # 2 = "Hold action"

		options = self.model.predict(state)  # Do a prediction based on a specific observation

		return np.argmax(options[0])'''

	def act(self, state):
		"""
		This function return NN model with "state_size" input and "action_size" output

		:param state: a tensor that represent a single observation of the environment

		:return: the action (prediction) with the best q-value  (the best action for this observation)
		"""
		rand_val = np.random.rand()
		if not self.is_eval and rand_val <= self.epsilon:  # Do a random action only in train phase
			return random.randrange(self.action_size)

		if self.firstIter:  # If this is the first iteration, just do a "hold" action
			self.firstIter = False
			return 2  # 2 = "Hold action"

		options = self.model.predict(state)  # Do a prediction based on a specific observation
		#print(options)

		tot = np.sum(options[0])
		options[0] = options[0] / tot
		#print(options)

		rand = random.random()

		#print("randm:" + str(rand))
		if rand <= options[0][0]:
			#print("max:" + str(np.argmax(options[0])) + "ma 0")
			return 0

		elif options[0][0] < rand <= (options[0][0] + options[0][1]):
			#print("max:" + str(np.argmax(options[0])) + "ma 1")
			return 1
		elif (options[0][0] + options[0][1]) < rand <= (options[0][0] + options[0][1] + options[0][2]):
			#print("max:" + str(np.argmax(options[0])) + "ma 2")
			return 2
		else:
			#print("max:" + str(np.argmax(options[0])) + "ma 3")
			return 3

		#return np.argmax(options[0])'''

	'''def exp_replay(self, batch_size):
		"""
		This method return NN model with "state_size" input and "action_size" output

		:param batch_size: the number of states to analyze for getting the "experience" for each time
		"""
		mini_batch = []
		memory_size = len(self.memory)  # Getting the memory size used for store the "experience"
		for i in range(memory_size - batch_size + 1, memory_size):
			mini_batch.append(self.memory.popleft())  # Loading the tuple (s, a, r, s')

		for state, action, reward, next_state in mini_batch:  # For each tuple of the "experience"
			#  Applying the Bellman Equation to compute the expected reward
			target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

			target_f = self.model.predict(state)  # Get the best action to do given a specific state
			target_f[0][action] = target  # Update the value of the original action with the best q-value
			result = self.model.fit(state, target_f, epochs=1, verbose=0)  # Update our NN-Agent
			print(result.history['loss'])
			print(type(result.history['loss']))

		if self.epsilon > self.epsilon_min:  # If we hadn't reached the minimum epsilon-random probability
			self.epsilon *= self.epsilon_decay  # Decrease the epsilon-random probability'''

	def exp_replay(self, batch_size):
		"""
		This method return NN model with "state_size" input and "action_size" output

		:param batch_size: the number of states to analyze for getting the "experience" for each time
		"""
		mini_batch = []
		memory_size = len(self.memory)  # Getting the memory size used for store the "experience"
		for i in range(memory_size - batch_size + 1, memory_size):
			mini_batch.append(self.memory.popleft())  # Loading the tuple (s, a, r, s')

		for state, action, reward, next_state in mini_batch:  # For each tuple of the "experience"
			#  Applying the Bellman Equation to compute the expected reward
			target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

			target_f = self.model.predict(state)  # Get the best action to do given a specific state
			target_f[0][action] = target  # Update the value of the original action with the best q-value
			result = self.model.fit(state, target_f, epochs=1, verbose=0)  # Update our NN-Agent
			self.loss += result.history['loss'][0]

		#print(np.divide(np.sum(results), 32))

		if self.epsilon > self.epsilon_min:  # If we hadn't reached the minimum epsilon-random probability
			self.epsilon *= self.epsilon_decay  # Decrease the epsilon-random probability

	def reset(self):
		"""
		This method reset the mini-batch memory

		"""
		self.memory = deque(maxlen=1000)  # Make a fast list push-pop
		self.loss = 0


