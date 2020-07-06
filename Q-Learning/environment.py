import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from PIL import Image
from playsound import playsound


class Environment:

    def __init__(self):
        self.field = np.zeros((7, 7))  # Creating our 7x7 field
        self.rat_position_x = 0  # Setting rat initial position on the 'x' axe
        self.rat_position_y = 0  # Setting rat initial position on the 'y' axe

        self.field[0][4] = self.field[2][0] = self.field[2][4] = self.field[3][6] = self.field[4][2] = self.field[5][5] = self.field[6][3] = 1  # Setting the holes
        self.field[2][1] = self.field[3][3] = self.field[4][1] = self.field[6][1] = 2  # Setting the light traps
        self.field[6][4] = 4  # Setting the cheese

        self.available_actions = ['up', 'down', 'left', 'right']
        self.life_time = 40  # The max number of timesteps in a episode before the mouse die of starvation
        self.step_cost = 1  # The reward cost of each move
        self.death = False  # Track if the mouse is death or not
        self.done = False  # Track if the episode is finished or not
        self.reward = 0  # # Track the total reward of the single episode
        self.journey = [(0, 0)]  # Store the historical moves of the mouse
        self.trigger_chance_of_trap = 0.5  # Set the trigger chance of the trap when meet the mouse

    def get_available_actions(self):
        # Return all available actions that we can do in the environment
        return self.available_actions

    def get_actions_n(self):
        # Return the total number of all available actions that we can do in the environment
        return len(self.available_actions)

    def get_state_size(self):
        n_row = self.field.shape[0]
        n_col = self.field.shape[1]
        return n_row, n_col

    def step(self, action):
        # Make a move following the action in the current system state, returning back the new state, the reward
        # and if the episode is ended or not

        self.reward = -self.step_cost  # The cost of doing a single step
        self.life_time -= self.step_cost  # Decrease the total life of the rat

        if action == "up":
            if not self.rat_position_y == 0:  # if we are not on high border
                self.rat_position_y -= 1  # Moving up
        elif action == "down":
            if not self.rat_position_y == 6:  # if we are not on lower border
                self.rat_position_y += 1  # Moving down
        elif action == "left":
            if not self.rat_position_x == 0:  # if we are not on left border
                self.rat_position_x -= 1  # Moving to the left
        elif action == "right":
            if not self.rat_position_x == 6:  # if we are not on right border
                self.rat_position_x += 1  # Moving to the right

        self._check_the_spot()
        self.journey.append((self.rat_position_y, self.rat_position_x))

        return (self.rat_position_y, self.rat_position_x), self.reward, self.done

    def render(self):
        # Show a playback of the current episode
        journey_len = len(self.journey) - 1
        for index, coord in enumerate(self.journey):
            if index == journey_len:  # If we are at the last move
                if coord == (6, 4):  # If we had reach the cheese, we had won!
                    image_path = "images/alive/field_y{}_x{}.png".format(coord[0], coord[1])
                else:  # If the last move is on a trap...see you at hell!
                    image_path = "images/death/field_y{}_x{}.png".format(coord[0], coord[1])
                    if self.field[coord[0], coord[1]] == 2:
                        playsound('sound/Shock.mp3')
                    else:
                        playsound('sound/Whilelm_scream.wav')
            else:
                image_path = "images/alive/field_y{}_x{}.png".format(coord[0], coord[1])

            im = Image.open(image_path)
            plt.figure(figsize=(16, 9))
            plt.imshow(im)
            plt.title("On going experiment..")
            plt.pause(1)

            if index == journey_len and coord == (6, 4):  # If we had reach the cheese, we had won!
                playsound('sound/Victory.wav')

            plt.close()

        plt.pause(5)

    def reset(self):
        # Reset the environment in order to run a new episode
        self.rat_position_x = 0  # Setting the rat initial position
        self.rat_position_y = 0  # Setting the rat initial position
        self.life_time = 40
        self.death = False
        self.done = False
        self.journey = [(0, 0)]  # Store the historical moves of the mouse
        return self.rat_position_y, self.rat_position_x

    def _check_the_spot(self):
        if self.field[self.rat_position_y, self.rat_position_x] == 1:  # If there's a hole, have a nice trip my buddy!
            self._rip_mouse()
            print("The subject fell down through the hole!")
        elif self.field[self.rat_position_y, self.rat_position_x] == 2:  # If there's the trap..pray my buddy!
            if int(rnd.uniform(0, 1)) < self.trigger_chance_of_trap:  # Chance that the trap is activated
                self._rip_mouse()
                print("The trap's triggered! and cooked the subject..")
        elif self.field[self.rat_position_y, self.rat_position_x] == 4:
            self._victory()

        if self.life_time == 0 and not self.death:  # If the rat's life is ended, RIP old my buddy
            print("The subject die for age! RIP my old buddy...")
            self._rip_mouse()

    def _rip_mouse(self):
        self.death = True
        self.done = True
        self.reward = -100

    def _victory(self):
        self.done = True
        self.reward = 200
        print("The subject has reached the prize!")
