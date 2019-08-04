# import required libraries
import gym
import numpy as np


# function to generate discrete values from continuous values
def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int))


# creating the environment using OpenAI gym
env = gym.make('MountainCar-v0')

# initializing some parameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000
SHOW_EVERY = 100

# initializing values for the exploration and exploitation problem
epsilon = 0.5
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAY - START_EPSILON_DECAY)

# generating a 2D state space dimensions 20x20 (position, velocity)
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
# creating a window size for descretizing the state space
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# create a Q-table for state-action pairs (20x20x3)
# 3 actions possible here
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


# start of the training process
for episode in range(EPISODES):
	if episode%SHOW_EVERY == 0:
		print(episode)
		render = True
	else:
		render = False
	# convert the state to discrete
	discrete_state = get_discrete_state(env.reset())
	# check if the eposide is completed
	done = False
	while not done:
		# take a random action or action with maximum q-value
		if np.random.random() > epsilon:
			action = np.argmax(q_table[discrete_state])
		else:
			action = np.random.randint(0, env.action_space.n)
		# taking an action and output is a new_state and reward
		new_state, reward, done, _ = env.step(action)
		new_discrete_state = get_discrete_state(new_state)
		# display the environment
		if render:
			env.render()
		# implementing the Q-Learning equation
		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action, )]
			# Q-Learning equation
			new_q = current_q + LEARNING_RATE * ((reward + DISCOUNT * max_future_q) - current_q)
			q_table[discrete_state + (action, )] = new_q
		elif new_state[0] >= env.goal_position:
			print(f'We made it to the goal location on episode {episode}')
			q_table[discrete_state + (action, )] = 0
		discrete_state = new_discrete_state
	# decrease the exploration as the number of episodes increases
	if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAY:
		epsilon -= epsilon_decay_value

# closing the environment
env.close()