import numpy as np
import pickle
from PIL import Image
import cv2
import matplotlib.pyplot as plt

SIZE = 10

TOT_EPISODES = 20001
SHOW = 2000

epsilon = 0.9
EPSILON_DECAY = 0.9

qTableStart = None #Start with None, but after the first cycle the saved one can be used writing the file name

MOVEMENT_PENALTY = 1
ENEMY_PENALTY = -300
SAFE_REWARD = 25

DISCOUNT = 0.95
LEARNING_RATE = 0.35

class Agent:
    def __init__(self, x=False, y=False):
        if x == False:
            self.x = np.random.randint(0, SIZE)
        else:
            self.x = x
        
        if y == False:
            self.y = np.random.randint(0, SIZE)
        else:
            self.y = y
    
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)
    
    #There are 4 possible moves: 0 == up, 1 == down, 2 == left, 3 == right
    def move(self, choice):
        if choice == 0:
            self.x += 0
            self.y += 1
        elif choice == 1:
            self.x += 0
            self.y += -1
        elif choice == 2:
            self.x += 1
            self.y += 0
        elif choice == 3:
            self.x += -1
            self.y += 0
        
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE - 1:
            self.x = SIZE - 1

        if self.y < 0:
            self.y = 0
        elif self.y > SIZE - 1:
            self.y = SIZE - 1




#Definition of the qTable

if qTableStart is None:
    qTable = {}
    #The state space is (x1, y1),(x2, y2), it's the delta between the player and the safe zone and the player and the enemy
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    qTable[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    qTable = pickle.load(open(qTableStart, "rb"))




#Learning cycle

episodeRewards = []

wins = 0

for episode in range(TOT_EPISODES):
    episodeReward = 0

    safeZone = Agent(5, 5)

    player = Agent()
    enemy = Agent()

    #To match the game's rules, neither the player nor the enemy can spawn in the safe zone
    while (player.x == safeZone.x and player.y == safeZone.y) or (enemy.x == safeZone.x and enemy.y == safeZone.y):
        player = Agent()
        enemy = Agent()

    for i in range(250):
        currentState = (player - safeZone, player - enemy)
        
        #With probability epsilon we choose a random move, otherwise the best move is selected
        if np.random.uniform(0, 1) > epsilon:
            move = np.argmax(qTable[currentState])
        else:
            move = np.random.randint(0,4)
        
        player.move(move)

        #The enemy moves randomly
        enemy.move(np.random.randint(0,4))

        #The player wins if it reaches the safe zone
        if player.x == safeZone.x and player.y == safeZone.y:
            reward = SAFE_REWARD
            wins += 1
        #The player loses if it is caught by the enemy or if the enemy gets to the safe zone first
        elif (player.x == enemy.x and player.y == enemy.y) or (enemy.x == safeZone.x and enemy.y == safeZone.y):
            reward = ENEMY_PENALTY
        #Every move has a small penalty
        else:
            reward = MOVEMENT_PENALTY
       
        #Q-Learning update
        nextState = (player - safeZone, player - enemy)
        bestNextAction = np.max(qTable[nextState])
        currentQ = qTable[currentState][move]
        newQ = currentQ + LEARNING_RATE*(reward + DISCOUNT*bestNextAction - currentQ)
        
        qTable[currentState][move] = newQ

        episodeReward +=reward
        
        #Showing the gameplay every SHOW episodes
        if episode % SHOW == 0:

            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[player.x][player.y] = (255, 0, 0) # The player is blue
            env[safeZone.x][safeZone.y] = (0, 255, 0) #The safe zone is green
            env[enemy.x][enemy.y] = (0, 0, 255) #The enemy is red
            
            img = Image.fromarray(env, "RGB")
            img = img.resize((500,500))
            cv2.imshow("", np.array(img))
            if reward == SAFE_REWARD or reward == ENEMY_PENALTY:
                if cv2.waitKey(1000) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
            
            if reward == SAFE_REWARD:
                print("Episode " + str(episode) + ": WIN") 
                break
            elif reward == ENEMY_PENALTY:
                print("Episode " + str(episode) + ": LOST")
                break
        
        if reward == SAFE_REWARD or reward == ENEMY_PENALTY:
            break

    episodeRewards.append(episodeReward)
    epsilon *=EPSILON_DECAY



#Printing the results

print("Number of wins in " + str(TOT_EPISODES) + " episodes: " + str(wins))


#The graph needs to be plotted using the moving average, showing the reward of every episode makes it impossible to read
movingAgerage = np.convolve(episodeRewards, np.ones((SHOW,))/SHOW, mode='valid')

plt.plot([i for i in range(len(movingAgerage))], movingAgerage)
plt.show()


#Saving the qTable 
pickle.dump(qTable, open("qTable-D" + str(DISCOUNT) + "LR" + str(LEARNING_RATE) + ".pickle", "wb"))