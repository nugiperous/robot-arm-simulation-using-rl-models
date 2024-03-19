import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        # initialize the replay buffer with the given maximum size, input shape and number of actions
        self.mem_size = max_size
        self.mem_cntr = 0
        # create numpy arrays to store the state, new state, action, reward, and terminal values for each transition
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        # store a transition into the replay buffer
        index = self.mem_cntr % self.mem_size # get the index of the oldest transition in the buffer
        self.state_memory[index] = state # store the current state
        self.action_memory[index] = action # store the action taken
        self.reward_memory[index] = reward # store the reward received
        self.new_state_memory[index] = state_ # store the new state
        self.terminal_memory[index] = done # store whether the new state is terminal or not

        self.mem_cntr += 1 # increase the counter for number of stored transitions

    def sample_buffer(self, batch_size):
        # sample a batch of transitions from the replay buffer
        max_mem = min(self.mem_cntr, self.mem_size) # get the number of stored transitions

        batch = np.random.choice(max_mem, batch_size) # randomly select a batch of transitions

        # extract the state, action, reward, new state, and terminal values for the selected transitions
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
