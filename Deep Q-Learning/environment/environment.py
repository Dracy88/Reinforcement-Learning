# ************************************************ Importing Libraries *************************************************
import pandas as pd
import numpy as np
import math


class Environment:

    def __init__(self, ds_path, window_size, pip_pos, stop_loss, trans_cost=0):
        """
        :param ds_path: specific the location path of the stock dataset
        :param window_size: specific the number of different prices in the same state
        :param pip_pos: specific the pip's position
        :param stop_loss: specific the maximum lost (in pips) that we can handle
        :param trans_cost: [Optional] specific the cost of a single transaction
        """
        self.actual_price = 0  # Contain the current price of a specific time step
        self.active_order_price = 0  # The price when the order has opened
        self.order_type = ""  # Specific if an order is 'long' or 'short' type
        self.is_active_order = False  # Monitor if there's an order active
        self.time_step = 0  # Contain the current instance index
        self.trans_cost = trans_cost  # The cost of transaction imposed by our broker

        self.pip_factor = math.pow(10, pip_pos)  # Needed to convert a differences of two prices in pips
        self.window_size = window_size  # The number of the close prices in a single state
        self.done = False  # Monitoring when we had reached the last instance
        self.stop_loss = stop_loss  # Setting the maximum loss that we can afford for a single order
        self.available_actions = ["Buy_long", "Buy_short", "Hold", "Close"]  # The available actions of the environment
        self.profit = 0  # The virtual profit; monitor the reward that we will get if we close an order
        self.n_feature = 2  # The number of different feature of our state (e.g. close, v_profit)

        self.ds_path = ds_path  # Setting the dataset path where we had all instances of prices
        self.ds = self._load_datas()  # A pandas data frame that contain all instances of the prices
        self.ds_len = len(self.ds)  # The number of instances of our dataset

        self.stop_loss_triggered = False  # Monitor if the stop loss system has triggered
        self.state = self._get_first_state()  # Contain the current state of a specific time step

        print("Environment Created")
        print("Window size:", self.window_size)

    def _load_datas(self):
        """
        This function load the dataset into a pandas data frame

        :return: a pandas data frame of the dataset
        """
        ds = pd.read_csv(self.ds_path, sep=',', header=0, dtype='float32')
        print("Founded", ds.shape[0], "instances")
        return ds

    def get_n_instances(self):
        return self.ds.shape[0]

    def step(self, action):
        """
        This function apply an action and manage the entire environment.
        :param action: the action choose from the external
        :return: a new state after the action has income and the reward obtained in a specific time step
        """
        if action not in self.available_actions:  # Checking if the entering action is valid
            raise ValueError(action, "is not a valid action")

        reward = 0
        self.time_step += 1
        self.profit = 0
        self.stop_loss_triggered = False

        # ************** If there's a Buy Order and there are not Other Active Orders, Open a new Order ****************
        if (action == "Buy_long" or action == "Buy_short") and not self.is_active_order:
            self.is_active_order = True  # We had an order active on the market
            self.order_type = action  # Saving if the order is "long" or "short" type
            self.active_order_price = self.actual_price  # Saving the price which we had open the order
        elif action == "Close" and self.is_active_order:
            self.is_active_order = False  # We don't had anymore an order active
            self.order_type = ""  # Resetting the order type
            reward = self.state[0][-1]  # Getting the last reward

        self.actual_price = self.get_last_price()  # Getting the last price

        # ************* Recalculate the Virtual Profit if there's an Order Active for this Specific State **************
        if self.is_active_order:  # If there's an active order on the market
            if self.order_type == "Buy_long":  # If the order is "long" we calculate the profit in this way
                # ***************************************** Stop Loss System *******************************************
                if ((self.ds['Low'].iloc[self.time_step] -
                     self.active_order_price) * self.pip_factor) <= self.stop_loss - self.trans_cost:
                    # If the actual v-profit has reached the max loss that we can handle, close the order
                    self.is_active_order = False
                    self.order_type = ""
                    self.profit = self.stop_loss - self.trans_cost
                    reward = self.profit
                    self.stop_loss_triggered = True  # Enable the stop_loss flag, so the trader can be notified
                else:
                    self.profit = ((self.actual_price - self.active_order_price) * self.pip_factor) - self.trans_cost

            else:  # Otherwise if the order is "short" we calculate the profit in another way
                # ***************************************** Stop Loss System *******************************************
                if ((self.active_order_price -
                     self.ds['High'].iloc[self.time_step]) * self.pip_factor) <= self.stop_loss - self.trans_cost:
                    # If the actual v-profit has reached the max loss that we can handle, close the order
                    self.is_active_order = False
                    self.order_type = ""
                    self.profit = self.stop_loss - self.trans_cost
                    reward = self.profit
                    self.stop_loss_triggered = True  # Enable the stop_loss flag, so the trader can be notified
                else:
                    self.profit = ((self.active_order_price - self.actual_price) * self.pip_factor) - self.trans_cost

        self.state = self.get_state()  # Obtain the last state generated by the action used

        if self.time_step + self.window_size - 1 > self.ds_len:  # If we had reach the last line of the dataset
            self.done = True

        return self.state, reward

    def get_actions(self):
        """
        This function return all the available action

        :return: a list of all the available actions
        """
        return self.available_actions

    def get_actions_n(self):
        """
        This function return the number of the available actions

        :return: the number of available actions
        """
        return len(self.available_actions)

    def get_state_size(self):
        """
        This function return the size of a single state, defined by the window size * 2 (one for the 'close' price,
            and another for 'virtual profit')

        :return: window_size * number of feature
        """
        return self.window_size * 2  # Return the size of a state

    def get_state(self):
        """
        This function get the current state

        :return: the new state based on the last action
        """
        prev_state = self.state  # Get the previous state (that before the current action)
        # Creating a new tensor of the same size of the previous state
        new_state = np.arange(0, self.window_size * self.n_feature, dtype=float)\
            .reshape(1, self.window_size * self.n_feature)
        # The new state is the previous (left shifted by n_feature) plus the new n_feature (e.g. close + v_profit)
        new_state[0][0: (self.window_size - 1)*self.n_feature] = prev_state[0][2:]
        new_state[0][-2] = self.actual_price  # The penultimate state cell contains the current price
        new_state[0][-1] = round(self.profit, 1)  # The last state cell contains the current v_profit

        return new_state

    def reset(self):
        """
        This method reset the entire environment to initial values, useful when we wanna play a new episode
        """
        self.actual_price = 0
        self.active_order_price = 0
        self.order_type = ""
        self.is_active_order = False
        self.time_step = 0
        self.done = False
        self.profit = 0
        self.state = self._get_first_state()
        print("Environment has been Resetted")

    def get_last_price(self):
        """
        This function return the price of the last state

        :return: the time_step-instance of the dataset
        """
        return self.ds['Close'].iloc[self.time_step]

    def _get_first_state(self):
        """
        This function make and return the first state of the environment

        :return: the first state of the environment
        """
        initial_close = self.ds['Close'].iloc[0: self.window_size]  # Getting the first "window_size" instances
        initial_close = (np.asanyarray(initial_close)).reshape(1, self.window_size)  # Reshaping into TF friendly format
        initial_profit = np.zeros((1, self.window_size))  # Setting the relative profits to a 0
        first_state = np.zeros((1, self.window_size * self.n_feature))  # Tensor that can contain prices and profits
        first_state[0][::self.n_feature] = initial_close[0]  # Insert the close prices in alternate cells
        first_state[0][1::self.n_feature] = initial_profit[0]  # Insert the initial virtual profit in alternate cells
        self.time_step = self.window_size - 1  # Setting the time step on the index of the next instances of data frame

        return first_state
