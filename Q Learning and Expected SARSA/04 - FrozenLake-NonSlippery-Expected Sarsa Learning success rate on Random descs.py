# OpenAI Gym non-slippery FrozenLake-v1 environment Q-Learning

import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map

total_num_of_random_descs = 10
return_values = []
success_rates_for_random_descs = []

# Run experiments on total_num_of_random_descs random descs
for desc in range(total_num_of_random_descs):
    print("Desc number: ", desc)
    
    # Initialize 4x4 non-slippery FrozenLake-v1 environment using gym
    # Initializes default 4x4 map
    # env = gym.make("FrozenLake-v1", is_slippery=False)
    # total_training_runs_on_a_desc, total_num_of_value_iterations = 50, 1000
    # Initializes the default 8x8 map
    # env = gym.make("FrozenLake-v1", is_slippery=False, map_name = "8x8")
    # total_training_runs_on_a_desc, total_num_of_value_iterations = 50, 3000
    # Initializes a random 8x8 map
    env = gym.make('FrozenLake-v1', is_slippery=False, desc=generate_random_map(size=8))
    (total_training_runs_on_a_desc, total_num_of_value_iterations) = (50, 5000)
    env.reset()

    # number of states
    num_states = env.observation_space.n
    # number of actions
    num_action = env.action_space.n
    terminal_state = num_states - 1

    # Initialize action-values
    q_values = np.zeros((num_states, num_action))

    epsilon = 0.01
    discount_factor = 0.9
    step_size = 0.5
    theta = 0.001
    return_values_for_a_desc = []


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
    def get_epsilon_greedy_action(state, current_epsilon):
        # If random float is less than epsilon, choose a random action
        if np.random.rand() < current_epsilon:
            return env.action_space.sample()
        # Return a greedy action
        return argmax_with_random_tie(q_values[state])


    # GPI: Value-iteration Q-Learning
    def value_iteration(current_epsilon):

        # Reset environment
        env.reset()

        is_terminal = False
        state = 0

        while not is_terminal:
            action = get_epsilon_greedy_action(state, current_epsilon)

            # Take action and get resultant next state, reward and termination info
            result = env.step(action)
            current_state = result[0]
            reward = result[1]
            is_terminal = result[2]
            
            if not is_terminal:
                q_values[state, action] = q_values[state, action] + step_size * (reward + discount_factor * np.average(q_values[current_state, :]) - q_values[state, action])
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
        
    # Training the agent
    def train_the_agent(total_iterations):
        # Initialize q_values
        q_values = np.zeros((num_states, num_action))
        for iteration in range(total_iterations):
            # print("Iteration: ", iteration+1)
            current_epsilon = 1 - (iteration / total_iterations)
            value_iteration(current_epsilon)
            # return_value = policy_evaluation()
            # print("Return: ", return_value)

    # Running the agent as per the learned policy...
    def is_policy_successful():
        # print("Running the agent...")
        env.reset()
        state = 0
        is_terminal = False
        num_of_steps = 0
        return_value = 0
        discount = 1

        while not is_terminal:
            action = policy(state)
            # print("Taking action ", action, " on state ", state)
            result = env.step(action)
            current_state = result[0]
            reward = result[1]
            num_of_steps = num_of_steps + 1
            is_terminal = result[2]
            return_value = return_value + discount * reward
            discount = discount * discount_factor
            
            # print("Reward = ", reward, " and current state = ", current_state)
            state = current_state
        
        return_values_for_a_desc.append(return_value)
        return (reward > 0)

    # Run experiments to test the training algo
    def run_experiments(total_training_runs_on_a_desc, total_num_of_value_iterations):
        passed_experiments = 0
        for run in range(total_training_runs_on_a_desc):
            train_the_agent(total_num_of_value_iterations)
            if is_policy_successful():
                passed_experiments = passed_experiments + 1
        
        return (100 * passed_experiments / total_training_runs_on_a_desc)
        

    # Run experiments to get the success rate of the training algo
    success_rate_for_desc = run_experiments(total_training_runs_on_a_desc, total_num_of_value_iterations)
    success_rates_for_random_descs.append(success_rate_for_desc)

    # Uncomment to print return_value for each experiment stored in return_values_for_a_desc
    print("Return value for each iteration for training:")
    print(return_values_for_a_desc)
    return_values.append(return_values_for_a_desc)

    # Print the result
    print("num of states of the desc: ", num_states)
    print("Success rate of the learnt policy for a random desc is ", success_rate_for_desc, "%!!!")

    env.close()

print("Return values:")
print(return_values)

avg_success_rate = np.average(np.array(success_rates_for_random_descs))
print("Average success rate = ", avg_success_rate)

print("Success rate history:")
print(success_rates_for_random_descs)

# Average success rate =  92.8