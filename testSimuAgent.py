import gym
import gym_simu
env = gym.make('simu-v0')

for _ in range(30):
    env.render()
    env.step(env.action_space.sample()) # take a random action
    # env.step(None) # take a random action

print("resetting")
env.reset()

for _ in range(30):
    env.render()
    env.step(env.action_space.sample()) # take a random action
    # env.step(None) # take a random action


env.close()