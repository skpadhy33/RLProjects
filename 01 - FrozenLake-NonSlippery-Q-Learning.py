# OpenAI Gym non-slippery FrozenLake-v1 environment Q-Learning

import numpy as np
import gym

# Initialize 4x4 non-slippery FrozenLake-v1 environment using gym
env = gym.make("FrozenLake-v1", is_slippery=False)
env.reset()

# number of states = 16
num_states = env.observation_space.n
# number of actions = 4
num_action = env.action_space.n
terminal_state = num_states - 1

# Initialize action-values
q_values = np.zeros((nb_states, nb_action))

epsilon = 0.05
discount_factor = 0.9
step_size = 0.1
theta = 0.001
return_values = []


# Returns argmax with random ties
def argmax_with_random_tie(values):
    argmax_values = []
    max_val = float("-inf")
    for i, val in enumerate(values):
        val = values[i]
        if val > max_val:
            argmax_values = [i]
            max_val = val
        elif val == max_val:
            argmax_values.append(i)
    return argmax_values[np.random.randint(low = 0, high = len(argmax_values))]

# returns an epsilon greedy action
def get_epsilon_greedy_action(state):
    # If random float is less than epsilon, choose a random action
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    # Return a greedy action
    return argmax_with_random_tie(q_values[state])


# GPI: Value-iteration Q-Learning
def value_iteration():

    # Reset environment
    env.reset()

    is_terminal = False
    state = 0

    while not is_terminal:
        action = get_epsilon_greedy_action(state)

        # Take action and get resultant next state, reward and termination info
        result = env.step(action)
        current_state = result[0]
        reward = result[1]
        is_terminal = result[2]
        
        if not is_terminal:
            q_values[state, action] = q_values[state, action] + step_size * (reward + discount_factor * np.max(q_values[current_state, :]) - q_values[state, action])
        else:
            q_values[state, action] = q_values[state, action] + step_size * (reward - q_values[state, action])
        
        state = current_state


# Policy function
def policy(state):
    return argmax_with_random_tie(q_values[state,:])


# Policy Evaluation
def policy_evaluation():
    env.reset()

    state = 0
    is_terminal = False
    return_value = 0.0
    discount = 1.0

    while not is_terminal:
        action = policy(state)
        result = env.step(action)
        state = result[0]
        reward = result[1]
        is_terminal = result[2]
        return_value = return_value + discount * reward
        discount = discount * discount_factor
    return return_value
    
for iteration in range(500):
    print("Iteration: ", iteration+1)
    value_iteration()
    return_value = policy_evaluation()
    print("Return: ", return_value)
    return_values.append(return_value)
    

# Running the agent...
print("Running the agent...")
env.reset()
state = 0
is_terminal = False
num_of_steps = 0
return_value = 0
discount = 1

while not is_terminal:
    action = policy(state)
    print("Taking action ", action, " on state ", state)
    result = env.step(action)
    current_state = result[0]
    reward = result[1]
    num_of_steps = num_of_steps + 1
    is_terminal = result[2]
    return_value = return_value + discount * reward
    discount = discount * discount_factor
    
    print("Reward = ", reward, " and current state = ", current_state)
    state = current_state

print("Return for the policy = ", return_value, " total steps = ", num_of_steps)

print(return_values)

env.close()