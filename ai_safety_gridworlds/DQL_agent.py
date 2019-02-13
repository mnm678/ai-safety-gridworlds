import numpy as np
import tensorflow as tf
import random
import collections
import sys
import time

from ai_safety_gridworlds.environments.shared.safety_game import Actions
from ai_safety_gridworlds.helpers import factory

actions = [Actions.RIGHT, Actions.UP, Actions.DOWN, Actions.LEFT]
possible_actions = np.array(np.identity(len(actions),dtype=int).tolist())

random.seed(5)

state_size = [6, 8, 4] #4 stacked frames
total_state_size = state_size[0] * state_size[1] * state_size[2]
action_size = len(possible_actions)
#learning rate
alpha = 0.0035

#Training hyperparameters
total_episodes = 300
max_steps = 50
batch_size = 700 

#exploration parameters
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0005

#q learning
gamma = 0.95

#memory hyperparameters
pretrain_length = batch_size
memory_size = 100000

training = True

print(time.time())

stack_size = 4

stacked_frames  =  collections.deque([np.zeros((state_size[0], state_size[1]), dtype=np.float32) for i in range(stack_size)], maxlen=4)


def stack_frames(stacked_frames, state, is_new_episode):
    if is_new_episode:
        stacked_frames  =  collections.deque([np.zeros((state_size[0], state_size[1]), dtype=np.float32) for i in range(stack_size)], maxlen=4)
        stacked_frames.append(state)
        stacked_frames.append(state)
        stacked_frames.append(state)
        stacked_frames.append(state)
        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(state)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames


if __name__ == '__main__':
    num_args = len(sys.argv)
    if num_args > 1:
        alpha = float(sys.argv[1])
    if num_args > 2:
        decay_rate = float(sys.argv[2])

class DQN:
    def __init__(self, state_size, action_size, learning_rate, name="DQN"):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.inputs = tf.placeholder(tf.float32, [None, state_size[0], state_size[1], state_size[2]], name="inputs")
            self.conv = tf.layers.conv2d(inputs = self.inputs,
                                         filters = 16,
                                         kernel_size = 1,
                                         strides = 1,
                                         padding = "VALID",
                                         name = "conv1")
            self.conv1_out = tf.nn.elu(self.conv, name="conv1_out")
            self.flatten = tf.contrib.layers.flatten(self.conv1_out)

            self.actions = tf.placeholder(tf.float32, [None, self.action_size], name="actions")
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            self.fc1 = tf.layers.dense(self.flatten, 50, activation=tf.nn.relu)
            #self.fc2 = tf.layers.dense(self.fc1, 50, activation=tf.nn.relu)

            self.output = tf.layers.dense(self.fc1, units=action_size)
            self.q = tf.reduce_sum(tf.multiply(self.output, self.actions))

            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.q))
            #self.optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(self.loss)
            self.optimizer = tf.train.AdamOptimizer(alpha).minimize(self.loss)


tf.reset_default_graph()
DQN = DQN(state_size, action_size, alpha)

class Queue:
    def __init__(self, size):
        self.queue = collections.deque(maxlen = size)
    def add(self, item):
        self.queue.append(item)
    def sample(self, batch_size):
        queue_size = len(self.queue)
        index = np.random.choice(np.arange(queue_size),
                                size = queue_size,
                                replace = False)
        return [self.queue[i] for i in index]
    def printQueue(self):
        contents = ", ".join(map(str, self.queue))
        return "Queue[{}]".format(contents)

queue = Queue(memory_size)


def initialize():
    env = factory.get_environment_obj('island_navigation')
    #hack to get the initial boad using a noop
    timestep = env.step(Actions.NOOP)
    state = timestep.observation["board"]

    global stacked_frames
    state, stacked_frames = stack_frames(stacked_frames, state, True)

    reward = 0
    for i in range(pretrain_length):
        #do random action
        action_idx = random.randint(0,action_size-1)
        action = possible_actions[action_idx]
        timestep = env.step(actions[action_idx])
        reward += timestep.reward if timestep.reward else 0
        done = timestep.last()

        if done:
            next_state = np.zeros((state_size[0], state_size[1]), dtype=np.float32)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            if(reward > 0):
                for i in range(0, 20):
                    queue.add((state, action, timestep.reward, next_state, done))
            queue.add((state, action, timestep.reward, next_state, done))
            env.reset()
            #this hack again
            timestep = env.step(Actions.NOOP)
            state = timestep.observation["board"]
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            reward = 0
        else:
            next_state = timestep.observation["board"]
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            if(reward > 0):
                for i in range(0, 20):
                    queue.add((state, action, timestep.reward, next_state, done))
            queue.add((state, action, timestep.reward, next_state, done))
            state = next_state

    #print(queue.printQueue())

initialize()

env = factory.get_environment_obj('island_navigation')


#pick best known or exploration
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions, sess):
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        action_idx = random.randint(0,action_size-1)
        action = possible_actions[action_idx]

    else:
        Qs = sess.run(DQN.output, feed_dict = {DQN.inputs: state.reshape((1, state_size[0], state_size[1], state_size[2]))})
        choice = np.argmax(Qs)
        action_idx = int(choice)
        action = possible_actions[action_idx]

    return action_idx, explore_probability


saver = tf.train.Saver()

scores = []

if training == True:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        decay_step = 0

        for episode in range(total_episodes):
            step = 0
            episode_rewards = []
            env.reset()
            #hack
            timestep = env.step(Actions.NOOP)
            state = timestep.observation['board']
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            reward = 0

            while step < max_steps:
                step += 1
                decay_step += 1
                action_idx, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions, sess)
                action = possible_actions[action_idx]

                timestep = env.step(actions[action_idx])
                reward += timestep.reward if timestep.reward else 0
                done = timestep.last()

                if done:
                    episode_rewards.append(reward)
                    scores.append(reward)
                    next_state = np.zeros((state_size[0], state_size[1]), dtype=np.float32)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    step = max_steps
                    total_reward = np.sum(episode_rewards)
                    if(reward > 0):
                        for i in range(0, 20):
                            queue.add((state, action, timestep.reward, next_state, done))
                    queue.add((state, action, timestep.reward, next_state, done))

                else:
                    next_state = timestep.observation['board']
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    if(reward > 0):
                        for i in range(0, 20):
                            queue.add((state, action, timestep.reward, next_state, done))
                    queue.add((state, action, timestep.reward, next_state, done))
                    state = next_state

                #do the learning periodically
                if done or (step % 10 == 0):

                    #learning
                    batch = queue.sample(batch_size)
                    states_mb = np.array([each[0] for each in batch], ndmin=3)
                    actions_mb = np.array([each[1] for each in batch])
                    rewards_mb = np.array([each[2] for each in batch])
                    next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                    dones_mb = np.array([each[4] for each in batch])

                    q_s_a = sess.run(DQN.output, feed_dict={DQN.inputs: states_mb})
                    q_s_a_d = sess.run(DQN.output, feed_dict={DQN.inputs: next_states_mb})

                    x = states_mb
                    y = np.zeros((len(batch)))
                    z = actions_mb

                    current_q = []
                    for i in range(0, len(batch)):
                        terminal = dones_mb[i]

                        # If we are in a terminal state, only equals reward
                        if terminal:
                            current_q.append(rewards_mb[i])

                        else:
                            current_q.append(rewards_mb[i] + gamma * np.max(q_s_a_d[i]))
                    #sess.run(DQN.optimizer, feed_dict={DQN.inputs: x, DQN.q: y})
                    y = current_q
                    sess.run([DQN.loss,DQN.optimizer], feed_dict={DQN.inputs: x, DQN.target_Q: y, DQN.actions: z})


            if episode % 10 == 0:
            #if episode == 10:
                save_path = saver.save(sess, "./models/model.ckpt")
print(scores)
#print(queue.printQueue())

config = tf.ConfigProto(device_count={'GPU': 2})
with tf.Session(config=config) as sess:
    env.reset()
    totalScore = 0

    saver.restore(sess, "./models/model.ckpt")

    for i in range(7):
        env.reset()
        #hack to get the initial boad using a noop
        timestep = env.step(Actions.NOOP)
        state = timestep.observation["board"]
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        score = 0
        while(not timestep.last()):
            state = timestep.observation["board"]
            state, stacked_frames = stack_frames(stacked_frames, state, False)
            Qs = sess.run(DQN.output, feed_dict = {DQN.inputs: state.reshape((1, state_size[0], state_size[1], state_size[2]))})
            action = np.argmax(Qs)
            action = actions[int(action)]
            timestep = env.step(action)
            score += timestep.reward if timestep.reward else 0
        print("Score: ")
        print(score)
        totalScore += score
    print("Total score: ")
    print(totalScore)

print(time.time())
