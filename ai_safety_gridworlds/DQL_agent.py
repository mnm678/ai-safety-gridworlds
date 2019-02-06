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
print(possible_actions)

state_size = [6, 8]
total_state_size = state_size[0] * state_size[1]
action_size = len(possible_actions)
#learning rate
alpha = 0.0001

#Training hyperparameters
total_episodes = 3000
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
            self.inputs = tf.placeholder(tf.float32, [None, state_size[0] * state_size[1]], name="inputs")
            self.q = tf.placeholder(tf.float32, [None, action_size])

            self.fc1 = tf.layers.dense(self.inputs, 50, activation=tf.nn.relu)
            self.fc2 = tf.layers.dense(self.fc1, 50, activation=tf.nn.relu)

            self.output = tf.layers.dense(self.fc2, action_size)

            self.loss = tf.losses.mean_squared_error(self.q, self.output)
            self.optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(self.loss)


tf.reset_default_graph()
DQN = DQN(state_size, action_size, gamma)

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

    reward = 0
    for i in range(pretrain_length):
        #do random action
        action = random.choice(possible_actions)
        timestep = env.step(action)
        reward += timestep.reward if timestep.reward else 0
        done = timestep.last()

        if done:
            next_state = np.zeros(state.shape)
            if(reward > 0):
                for i in range(0, 20):
                    queue.add((state.copy(), action, timestep.reward, next_state.copy(), done))
            queue.add((state.copy(), action, timestep.reward, next_state.copy(), done))
            env.reset()
            #this hack again
            timestep = env.step(Actions.NOOP)
            state = timestep.observation["board"]
            reward = 0
        else:
            next_state = timestep.observation["board"]
            if(reward > 0):
                for i in range(0, 20):
                    queue.add((state.copy(), action, timestep.reward, next_state.copy(), done))
            queue.add((state.copy(), action, timestep.reward, next_state.copy(), done))
            state = next_state.copy()

    #print(queue.printQueue())

initialize()

env = factory.get_environment_obj('island_navigation')


#pick best known or exploration
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions, sess):
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        action = actions[random.choice(possible_actions)]

    else:
        Qs = sess.run(DQN.output, feed_dict = {DQN.inputs: state.reshape((1, state_size[0] * state_size[1]))})
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_probability


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
            reward = 0

            while step < max_steps:
                step += 1
                decay_step += 1
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions, sess)

                timestep = env.step(action)
                reward += timestep.reward if timestep.reward else 0
                done = timestep.last()

                if done:
                    episode_rewards.append(reward)
                    scores.append(reward)
                    next_state = np.zeros(state.shape)
                    step = max_steps
                    total_reward = np.sum(episode_rewards)
                    if(reward > 0):
                        for i in range(0, 20):
                            queue.add((state.copy(), action, timestep.reward, next_state.copy(), done))
                    queue.add((state.copy(), action, timestep.reward, next_state.copy(), done))

                else:
                    next_state = timestep.observation['board']
                    if(reward > 0):
                        for i in range(0, 20):
                            queue.add((state.copy(), action, timestep.reward, next_state.copy(), done))
                    queue.add((state.copy(), action, timestep.reward, next_state.copy(), done))
                    state = next_state.copy()

                #do the learning periodically
                if done or (step % 10 == 0):
                    print("run start")
                    print(time.time())

                    #learning
                    batch = queue.sample(batch_size)
                    states_mb = np.array([each[0] for each in batch], ndmin=3)
                    actions_mb = np.array([each[1] for each in batch])
                    rewards_mb = np.array([each[2] for each in batch])
                    next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                    dones_mb = np.array([each[4] for each in batch])

                    q_s_a = sess.run(DQN.output, feed_dict={DQN.inputs: states_mb.reshape((-1, total_state_size))})
                    q_s_a_d = sess.run(DQN.output, feed_dict={DQN.inputs: next_states_mb.reshape((-1, total_state_size))})

                    x = np.zeros((len(batch), state_size[0] * state_size[1]))
                    y = np.zeros((len(batch), action_size))

                    for i in range(0, len(batch)):
                        current_q = q_s_a[i]
                        terminal = dones_mb[i]

                        # If we are in a terminal state, only equals reward
                        if terminal:
                            current_q[actions_mb[i]] = rewards_mb[i]

                        else:
                            current_q[actions_mb[i]] = rewards_mb[i] + gamma * np.max(q_s_a_d[i])
                        x[i] = states_mb[i].reshape(total_state_size)
                        y[i] = current_q
                    sess.run(DQN.optimizer, feed_dict={DQN.inputs: x, DQN.q: y})
                    print(time.time())


            if episode % 10 == 0:
            #if episode == 10:
                save_path = saver.save(sess, "./models/model.ckpt")
            print(time.time())
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
        score = 0
        while(not timestep.last()):
            state = timestep.observation["board"]
            Qs = sess.run(DQN.output, feed_dict = {DQN.inputs: state.reshape((1, state_size[0] * state_size[1]))})
            action = np.argmax(Qs)
            action = possible_actions[int(action)]
            timestep = env.step(action)
            score += timestep.reward if timestep.reward else 0
        print("Score: ")
        print(score)
        totalScore += score
    print("Total score: ")
    print(totalScore)

print(time.time())
