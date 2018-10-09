
# coding: utf-8

# In[327]:


import numpy as np

from keras.models import model_from_json
from keras.models import Sequential
from keras.optimizers import sgd
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
from keras.layers import RNN,SimpleRNN,LSTM,LSTMCell


import json
#matplotlib for rendering
import matplotlib.pyplot as plt
#numpy for handeling matrix operations
import numpy as np
#time, to, well... keep track of time
import time
#Python image libarary for rendering
from PIL import Image
#iPython display for making sure we can render the frames
from IPython import display
#seaborn for rendering
import seaborn



# In[328]:


last_frame_time = 0

gamma = .7
king_batch = 5
get_ipython().magic(u'matplotlib inline')
game_try = 10
epsilon=.1
fault = 0.2
hidden_size = 100
training_epochs = 500
environment_=5


# In[329]:


class environment:#maze environment
    def __init__(self,size):
        self.size = size
        self.actions = ['left','right','up','down']
        self.agent_position = [3,1]
        self.target_position = [size-1,size-1]
        self.state = np.zeros((size,)*2)
        self.reset()
        self.memory= set()
        
    def reset(self,batch=False):
#         self.agent_position = [np.random.randint(0, self.size-1 ,size=1)[0],np.random.randint(0, self.size-1 ,size=1)[0]]#random position on the maze
#         self.target_position = [np.random.randint(0, self.size-1 ,size=1)[0],np.random.randint(0, self.size-1 ,size=1)[0]]#random position on the maze
        self.agent_position = [0,1]
        self.memory= set()
        self.target_position = [self.size-1,self.size-1]
        
    def display_env(self):
        canvas = np.zeros((self.size,)*2)
        canvas[self.agent_position[0],self.agent_position[1]] = 1
        canvas[self.target_position[0],self.target_position[1]] = 1
        print(canvas)
        
    def observe(self):
        canvas = self.get_state()
        return canvas.reshape((1, -1))
        
    def move(self,action):
        #action is 1 for left -1 for right
        reward = 0
        action_y=0
        action_x=0
        
        position = np.copy(self.agent_position)
        
        if (position[0],position[1]) in self.memory:
#             print(self.memory)
            reward = reward+(-0.15)#discourage the niga from goin to the same place twice
        else:
            self.memory.add((position[0],position[1]))
        
        if(action== 0):
            action_y =  1
            
        elif(action== 1):
            action_y = (-1)
            
        elif(action== 2):
            action_x =  1
            
        else:
            action_x = (-1)
                
        done =False
          
        new_x = self.agent_position[0]+action_x 
        new_y = self.agent_position[1]+action_y 
        
        if(new_x == -1 or new_x == self.size):
            reward =  -0.3 +reward 
            
        if(new_y == -1 or new_y == self.size):
            reward = -0.3 + reward 
            
        new_x = min(self.size-1,max(0,new_x))
        new_y = min(self.size-1,max(0,new_y))
        
        self.agent_position = [new_x,new_y]
        
        if(self.agent_position ==   self.target_position):
            reward =  10
            done = True
#         elif(self.agent_position==[0,0]):# i want to discourage it from being at position 0
#             reward =  reward + -1
#             done=True
#             pass
        
        return reward,done,self.observe()
    
    def get_state(self):
#         print(self.agent_position)
        canvas = np.zeros((self.size,)*2)
        canvas[self.agent_position[0],self.agent_position[1]] = 1
        canvas[self.target_position[0],self.target_position[1]] = 1
        return canvas


# In[330]:


def display_screen(input_t):
    #Function used to render the game screen
    #Get the last rendered frame
    global last_frame_time
    #Only display the game screen if the game is not over
    plt.imshow(input_t.reshape((env.size,)*2),
    interpolation='none', cmap='gray')
    #Clear whatever we rendered before
    display.clear_output(wait=True)
    #And display the rendering
    display.display(plt.gcf())
    #Update the last frame time
    last_frame_time = set_max_fps(last_frame_time)
    
    
def set_max_fps(last_frame_time,FPS = 1):
    current_milli_time = lambda: int(round(time.time() * 1000))
    sleep_time = 1./FPS - (current_milli_time() - last_frame_time)
    if sleep_time > 0:
        time.sleep(sleep_time)
    return current_milli_time()


# In[331]:


class memory:
    
    def __init__(self,size=3000):
        self.max_size =  size
        self.memory = []
        
    def add(self,obj):
        """
        adding object [[S A R s ],game_end]]
        """
        self.memory.append(obj)
        
        if len(self.memory) > self.max_size:
            del self.memory[0]
            
    def get_a_training_batch(self,batch_size=king_batch):
        #am meant to get training batch and return inputs and training data
        le = len(self.memory)
        ran = np.random.randint(0, le ,size=min(le,batch_size))#getting random batches from the
        
        env_dim = self.memory[0][0][0].shape[1]
        
        inputs = np.zeros((batch_size,env_dim))
        
        target = np.zeros((batch_size,len(env.actions)))
        
        for i,item in enumerate(ran):
            
            #now i have random items in the arr eg [2,4,5,9]
            g=self.memory[item][0]
            state,action,reward,next_state = g
            inputs[i] = state
            #make a prediction with what we have now
            
            target[i] = model.predict(state)
            action = np.argmax(target[i])
            
            q=0
            if self.memory[item][1]==True:#means game ended
                #q = reward since no next action
                    q = reward
            #make a move and se if look for a reward
            else:
                q = reward  + gamma * np.max(model.predict(next_state)[0])
            target[i,action] = q
#         print(inputs,target)
        return inputs,target
            
            


# In[332]:


def train():
    mem = memory()
    avg_training_time = []
    avg_loss = []
    batch_loss=0
    maincoint=0
    totalwins=[]
    accuracy = 0
    totalloss = []
    wins = 0
    while wins < 400:
        wins = 0
        for i in range(training_epochs):

            done = False
            env.reset()#reset it after every game

            count=0
            
            loss = 0
            batch_loss=0
            print('start new game')
            agent_reward = 0
            while done==False:#wait till game is done
                input_state = env.observe()

                if len(mem.memory)>0:
                    inputs,target = mem.get_a_training_batch()

                    batch_loss += model.train_on_batch(inputs, target)

                if(np.random.rand() <=epsilon):
                    c = np.random.randint(0, len(env.actions), size=1)
                    action = c[0]
    #                 action = np.argmax(model.predict(input_state[np.newaxis])[0])

                else:
    #                 p= model.predict(input_state[np.newaxis])[0]
                    z = model.predict(input_state)[0]
                    action = np.argmax(z)

               


                reward,done_,_state = env.move(action)

                agent_reward+=reward

                mem.add([[input_state,action,reward,_state],done])


                count+=1

                if agent_reward < -fault*env.size:# monitors if it gets stucked
                    done_=True
#                     print('loss -__-)')
                    loss += 1
    #             print(env.get_state())
                elif done_ == True:
                    print('win!!!!')
                    wins += 1
                    
                done = done_
            print("end game")
            totalwins.append(wins)
            totalloss.append(loss)
            maincoint+=1
            print('maincount',maincoint)
            avg_loss.append(batch_loss/count)
            avg_training_time.append(count)
        print(wins , 'wins')
        accuracy = wins/training_epochs
        print('accuracy',accuracy)
    plt.plot(avg_loss)
    plt.ylabel('Average of loss per game over time')
    plt.show()
    
    plt.plot(totalloss)
    plt.ylabel('graph to show loss over time')
    plt.show()
    
    plt.plot(totalwins)
    plt.ylabel('graph to show game wins over time')
    plt.show()
    
    
    plt.plot(avg_loss)
    plt.ylabel('Average of training error per game over time')
    plt.show()
    
    plt.plot(avg_training_time)
    plt.ylabel('Average of training time ')
    plt.show()
    


# In[348]:


def baseline_model(grid_size,num_actions, lr=0.001):
    model = Sequential()
    model.add(Dense(80, input_shape=(grid_size**2,)))
    model.add(PReLU())
    model.add(Dense(50))
    model.add(PReLU())
    
#    model.add(Dense(90))
#     model.add(PReLU())
    
#     model.add(Dense(60))
#     model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model


# In[349]:


env = environment(environment_)
display_screen(env.get_state())





# In[350]:


#Define model

model = baseline_model(env.size,len(env.actions),hidden_size)
model.summary()


# In[351]:


def test():
    mem = memory()
    
    for i in range(game_try):
        done = False
        env.reset()
        while done == False:
            input_state = env.observe()
            action = np.argmax(model.predict(input_state)[0])
                
            reward,done_,_state = env.move(action)
            print(_state)
            
            display_screen(env.get_state())
           
            
            done = done_
            
            
            
    pass


# In[ ]:


train()


# In[ ]:


test()

