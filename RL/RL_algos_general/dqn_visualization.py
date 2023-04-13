### TEST MY NN on the env with visualization
import glob
import io
import base64
import ipythondisplay
import gym
import torch

def show_video():
  '''Enables video recording of gym environment and shows it.'''
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else:
    print("Video not found")

def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env


env = wrap_env(gym.make('CartPole-v1'))
state = env.reset()
done = False
ep_rew = 0
while not done:
  env.render()
  state = state.astype(np.float32)
  state = torch.from_numpy(np.expand_dims(state, axis=0)).to(device)
  action = select_epsilon_greedy_action(state, epsilon=0.01)
  state, reward, done, info = env.step(action)
  ep_rew += reward
print('Return on this episode: {}'.format(ep_rew))
env.close()
show_video()
