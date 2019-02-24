import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
import neuralnet
from operator import attrgetter

import time
import math
pygame.init()
np.set_printoptions(threshold=np.nan)

clock = pygame.time.Clock()
random.seed(7)
GRIDLENGTH = 20
GRIDHEIGHT = 20

WINDOWLENGTH = 400
WINDOWHEIGHT = 400
WINDOW_SIZE = [WINDOWLENGTH, WINDOWHEIGHT]

BATCHSIZE = 150
NUMGENERATIONS = 10000
tick = 0
tickrate = 60
ticklimit = 99


input_dim = 7      # 5 inputs - angle, distance, left, middle, right surroundings
hidden_dim = 100     # hidden units
output_dim = 3      # 0: Left, 1: Straight, 2: Right

# Select the top 25
select = 5
averageFitness = 0
fitnessStats = []
genStats = []
displayScreen = True
#Left: 1
#Right: -1
#Up: 2
#Down: -2

plt.figure(1000)
plt.title("Generation and fitness")
plt.ylabel("Fitness")
plt.xlabel("Generation")
# Function for generating the input vector that is fed to the NN
def calculateInput(grid, snake, fruit):

    # 1 is positive. 0 is negative
    leftFruit = 0
    straightFruit = 0
    behindFruit = 0
    rightFruit = 0

    leftClear = 0
    straightClear = 0
    rightClear = 0


    # Distance of snake from fruit
    distance = ((fruit.x - snake.x)**2 + (fruit.y - snake.y)**2)**(1/2)

    # Angle of snake from fruit (in radians)

    # If snake is moving left or right
    opp = 0
    # Distance of snake from fruit
    distance = ((fruit.x - snake.x)**2 + (fruit.y - snake.y)**2)**(1/2)
    headingX = 0
    headingY = 0
    if snake.direction == 1:
        headingX = -1
        headingY = 0

    elif snake.direction == -1:
        headingX = 1
        headingY = 0
    elif snake.direction == 2:
        headingY = -1
        headingX = 0
    elif snake.direction == -2:
        headingY = 1
        headingX = 0

    target = pygame.math.Vector2(-fruit.x+snake.x, -fruit.y+snake.y)
    heading = pygame.math.Vector2(headingX, headingY)
    product = float(pygame.math.Vector2.cross(target,heading))

    # Directly straight
    if math.copysign(1, product) == -1.0 and product == 0:
        straightFruit = 1
    # Directly Behind
    elif math.copysign(1, product) == 1.0 and product == 0:
        behindFruit = 1
    # Left
    elif product < 0:
        leftFruit = 1
    # Right
    elif product > 0:
        rightFruit = 1




    #angle = math.atan2(-(fruit.y-snake.y),(fruit.x-snake.x))
    #angle %= 2 * math.pi
    #print(angle*(180/math.pi))
    # # 1st quadrant (go left and straight)
    # if (angle > 0) and (angle < math.pi/2):
    #     leftFruit = 1
    #     straightFruit = 1
    # # 2nd quadrant (go left and behind)
    # if (angle > math.pi/2) and (angle < math.pi):
    #     leftFruit = 1
    #     behindFruit = 1
    # # 3rd quadrant (go right and behind)
    # if (angle > 0) and (angle < math.pi / 2):
    #     rightFruit = 1
    #     behindFruit = 1
    #
    # # 4th quadrant (go left)
    # if (angle > 0) and (angle < math.pi / 2):
    #     leftFruit = 1


    # Find what is directly to the left of
    left = 0
    straight = 0
    straights = np.ones(5)*3
    right = 0
    limit = 0
    iterY = 0
    iterX = 0
    angle = 0
    # Surroundings:
    # Now determine what is directly ahead of ahead of the snake (straight)
        # And what is left and right of it
    # Going left
    if snake.direction == 1:
        if grid.data[snake.x][snake.y+1] != 3 and grid.data[snake.x][snake.y+1] != 1:
            leftClear = 1
        if grid.data[snake.x-1][snake.y] != 3 and grid.data[snake.x-1][snake.y] != 1:
            straightClear = 1
        if grid.data[snake.x][snake.y-1] != 3 and grid.data[snake.x][snake.y-1] != 1:
            rightClear = 1

    # Right
    elif snake.direction == -1:
        if grid.data[snake.x][snake.y-1] != 3 and grid.data[snake.x][snake.y-1] != 1:
            leftClear = 1
        if grid.data[snake.x+1][snake.y] != 3 and grid.data[snake.x+1][snake.y] != 1:
            straightClear = 1
        if grid.data[snake.x][snake.y+1] != 3 and grid.data[snake.x][snake.y+1] != 1:
            rightClear = 1

    # Up
    elif snake.direction == 2:
        if grid.data[snake.x-1][snake.y] != 3 and grid.data[snake.x-1][snake.y] != 1:
            leftClear = 1
        if grid.data[snake.x][snake.y-1] != 3 and grid.data[snake.x][snake.y-1] != 1:
            straightClear = 1
        if grid.data[snake.x+1][snake.y] != 3 and grid.data[snake.x+1][snake.y] != 1:
            rightClear = 1

    # Down
    elif snake.direction == -2:
        if grid.data[snake.x+1][snake.y] != 3 and grid.data[snake.x+1][snake.y] != 1:
            leftClear = 1
        if grid.data[snake.x][snake.y+1] != 3 and grid.data[snake.x][snake.y+1] != 1:
            straightClear = 1
        if grid.data[snake.x-1][snake.y] != 3 and grid.data[snake.x-1][snake.y] != 1:
            rightClear = 1



    straight1 = straight
    ret = np.array([leftFruit,straightFruit,behindFruit,rightFruit,leftClear,straightClear,rightClear])
    #print(ret)
    #print(ret)
    return ret

def calculateFitness(net, grid, mySnake):
    fitness = mySnake.cumulativeDistance
    #explored = len(mySnake.history)/((GRIDLENGTH-2)*(GRIDHEIGHT-2))
    explored = 0
    fitness += explored*3 + mySnake.score
    #fitness += mySnake.score
    #print("The fitness is : " + str(fitness))
    return fitness

class Fruit:
    def __init__(self, grid):
        self.x = random.randint(0, GRIDLENGTH - 1)
        self.y = random.randint(0, GRIDHEIGHT - 1)
        self.sequence = [Coord(self.x, self.y)]
        self.seqnum = 0
        grid.data[self.x][self.y] = 2000

    def spawn(self, grid):

            # Use only if predictable fruit
            # self.x = self.sequence[0].x
            # self.y = self.sequence[0].y
            # self.seqnum = 0
            self.x = random.randint(1, GRIDLENGTH - 1)
            self.y = random.randint(1, GRIDHEIGHT - 1)
            while grid.data[self.x][self.y] != 0:
                self.x = random.randint(1, GRIDLENGTH - 1)
                self.y = random.randint(1, GRIDHEIGHT - 1)
            self.sequence.append(Coord(self.x, self.y))
            grid.data[self.x][self.y] = 2000

    def next(self, grid):

        # Use this if you want fruit to respawn at predictable locations.
        # # Next sequence
        # if self.seqnum < len(self.sequence)-1:
        #     self.seqnum+=1
        #
        #     self.x = self.sequence[self.seqnum].x
        #     self.y = self.sequence[self.seqnum].y
        # else:
        #     # Generate new fruit
        #     self.x = random.randint(1, GRIDLENGTH - 1)
        #     self.y = random.randint(1, GRIDHEIGHT - 1)
        #     while grid.data[self.x][self.y] != 0:
        #         self.x = random.randint(0, GRIDLENGTH - 1)
        #         self.y = random.randint(0, GRIDHEIGHT - 1)
        #     self.sequence.append(Coord(self.x,self.y))

        # Generate new fruit
        self.x = random.randint(1, GRIDLENGTH - 1)
        self.y = random.randint(1, GRIDHEIGHT - 1)
        while grid.data[self.x][self.y] != 0:
            self.x = random.randint(1, GRIDLENGTH - 1)
            self.y = random.randint(1, GRIDHEIGHT - 1)
        self.sequence.append(Coord(self.x, self.y))

        grid.data[self.x][self.y] = 2000


class Snake:

    # 0 = right
    # 1 = up
    # 2 = left
    # 3 = down
    def __init__(self, grid, fruit):
        self.x = GRIDLENGTH // 2
        self.y = GRIDHEIGHT // 2
        self.length = 1
        self.direction = random.choice((-1,1,2,-2))
        self.bannedDirection = -self.direction
        self.tail = [Coord(self.x,self.y)]

        self.history = [Coord(self.x,self.y)]

        #self.tail = []
        self.isGrowing = False
        
        self.score = 0
        self.highScore = 0
        self.dead = False
        self.life = 100
        self.cumulativeDistance = 0
        self.prevDistance = ((fruit.x - self.x)**2 + (fruit.y - self.y)**2)**(1/2)



    def respawn(self,grid, fruit):
        self.x = GRIDLENGTH // 2
        self.y = GRIDHEIGHT // 2
        self.length = 1
        self.direction = random.choice((-1,1,2,-2))
        self.bannedDirection = -self.direction
        self.tail = [Coord(self.x,self.y)]
        self.history = [Coord(self.x,self.y)]

        self.prevDistance = ((fruit.x - self.x)**2 + (fruit.y - self.y)**2)**(1/2)


        #self.tail = []
        self.isGrowing = False
        grid.clear(fruit)

        
    def update(self, grid, fruit):

        self.life -= 1
        # go left
        if self.direction == 1:
            self.x -= 1
        # go right
        if self.direction == -1:
            self.x += 1
        # go up
        if self.direction == 2:
            self.y -= 1
        # go down
        if self.direction == -2:
            self.y += 1


        # If collision with self or collision with wall, or starvation
        if (grid.data[self.x][self.y]) == 1 or (grid.data[self.x][self.y]) == 3 or self.life < 0:
            #print("dead. Score is : " + str(self.score) + ", high score is : " + str(self.highScore))
            self.dead = True

        distance = ((fruit.x - self.x) ** 2 + (fruit.y - self.y) ** 2) ** (1 / 2)
        if distance < self.prevDistance:
            self.cumulativeDistance += 0.0005
        else:
            self.cumulativeDistance -= 0.001
        self.prevDistance = distance
        # If fruit eaten
        if (grid.data[self.x][self.y]) == 2000:
            self.isGrowing = True
            fruit.next(grid)
            self.life += 100
            self.length += 1

            self.score += 1
            #print("eaten!")

        self.tail.append(Coord(self.x,self.y))

        isUniqueTile = True
        for coord in self.history:
            if coord.x == self.x and coord.y == self.y:
                isUniqueTile = False



        if isUniqueTile:
            self.history.append(Coord(self.x, self.y))

        #remove last member of tail. Don't do this if snake is growing.
        if (not self.isGrowing) and len(self.tail) > self.length:

            grid.data[self.tail[0].x][self.tail[0].y] = 0
            self.tail.pop(0)

        self.isGrowing = False
        #print("len is: " +str(self.length))
        # display all snake parts on grid
        for i in range(len(self.tail)):
            grid.data[self.tail[i].x][self.tail[i].y] = 1

class Coord:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Grid:
    def __init__(self):
        self.data = np.zeros((GRIDLENGTH, GRIDHEIGHT))
        self.width = 32
        self.height = 32
        self.thickness = 1

        # Set borders
        x = np.ones((0,GRIDLENGTH)) * 3
        y = np.ones((GRIDHEIGHT-1,0)) * 3
        self.data[:,0] = 3
        self.data[:,GRIDHEIGHT-1] = 3

        self.data[0,:] = 3
        self.data[GRIDLENGTH-1,:] = 3


    def clear(self,fruit):
        self.data = np.zeros((GRIDLENGTH, GRIDHEIGHT))

        # Set borders
        x = np.ones((0,GRIDLENGTH)) * 3
        y = np.ones((GRIDHEIGHT-1,0)) * 3
        self.data[:,0] = 3
        self.data[:,GRIDHEIGHT-1] = 3

        self.data[0,:] = 3
        self.data[GRIDLENGTH-1,:] = 3

        fruit.spawn(self)


screen = pygame.display.set_mode(WINDOW_SIZE)

# Initialize playing grid
# 0 = empty, 1 = player, 2 = fruit
grid = Grid()
# Set initial player at middle of screen

fruit = Fruit(grid)
snake = Snake(grid,fruit)

batch = []
# loading from gen 219
batch = np.load('savedGenerations\Generation_3049.npy')
batch = batch.tolist()
for i in range(BATCHSIZE):
    batch.append(neuralnet.NeuralNet())
newBatch = []

nn = neuralnet.NeuralNet()
netcounter = 0
# Main loop
done = False
currentgen = 3049
while not done:
    #for currentgen in range(NUMGENERATIONS):
    while currentgen < NUMGENERATIONS:
        print("RAW *******************RAW *******************RAW *******************RAW *******************RAW *******************")
        max = 0
        # Iterate through the entire batch
        for netNum in range(len(batch)):

            snake.dead = False
            tick = 0
            while not snake.dead:
                screen.fill([255, 255, 255])
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            done = True
                        if event.key == pygame.K_LEFT and snake.direction != -1:
                            snake.direction = 1
                        if event.key == pygame.K_RIGHT and snake.direction != 1:
                            snake.direction = -1
                        if event.key == pygame.K_UP and snake.direction != -2:
                            snake.direction = 2
                        if event.key == pygame.K_DOWN and snake.direction != 2:
                            snake.direction = -2


                        if event.key == pygame.K_z and snake.direction != -1:
                            tickrate*=2
                            print("Tickrate is now: " + str(tickrate))

                        if event.key == pygame.K_x and snake.direction != -1:
                            tickrate/=2
                            print("Tickrate is now: " + str(tickrate))

                        if event.key == pygame.K_a and snake.direction != -1:
                            ticklimit*=1.1
                            print("Ticklimit is now: " + str(ticklimit))
                        if event.key == pygame.K_s and snake.direction != -1:
                            ticklimit/= 1.1
                            print("Ticklimit is now: " + str(ticklimit))
                        if event.key == pygame.K_q:
                            displayScreen = not displayScreen

                        snake.bannedDirection = -snake.direction

                # Draw screen
                for i in range(grid.data.shape[0]):
                    for j in range(grid.data.shape[1]):
                        # background colour
                        color = [202,193,152]

                        # If grid is player
                        if grid.data[i, j] == 1:
                            color = [254,255,40]

                        # if grid is fruit
                        if grid.data[i, j] == 2000:
                            color = [255, 0, 0]

                        # if block is border
                        if grid.data[i, j] == 3:
                            color = [132,31,39]
                        if displayScreen:
                            pygame.draw.rect(screen, color, pygame.Rect(i * GRIDLENGTH, j * GRIDHEIGHT, grid.width, grid.height))
                        #print(grid.data)
                input = calculateInput(grid, snake, fruit)
                move = batch[netNum].forwardFeed(input)

                switchDirection = 0
                # Going left or right

               # print(move)
                # Go left
                if move == 0:
                    if snake.direction == -1:
                        snake.direction = 2
                    elif snake.direction == 1:
                        snake.direction = -2
                    elif snake.direction == -2:
                        snake.direction = -1
                    elif snake.direction == 2:
                        snake.direction = 1

                # Go right
                elif move == 2:
                    if snake.direction == -1:
                        snake.direction = -2
                    elif snake.direction == 1:
                        snake.direction = 2
                    elif snake.direction == -2:
                        snake.direction = 1
                    elif snake.direction == 2:
                        snake.direction = -1
                # Going straight ( move == 1) does nothing.
                snake.update(grid,fruit)
                #print(snake.dead)
                #print(tick)
                tick+=1
                clock.tick(tickrate)
                if (displayScreen):
                    pygame.display.flip()
            # Assign a fitness score to the network
            batch[netNum].fitness = calculateFitness(batch[netNum], grid, snake)
            if batch[netNum].fitness > max:
                max = batch[netNum].fitness
                print("New max is :" + str(max) + " at : " + str(netNum) + " in generation #: " + str(currentgen))
            averageFitness += batch[netNum].fitness
            newBatch.append(batch[netNum])
            snake = Snake(grid,fruit)
            netcounter+=1
            #print(snake.history)

           # print(str(batch[netNum].fitness))

            #print("NN# " + str(netcounter) + ". Generation " + str(currentgen) + ". Fitness " + str(batch[netNum].fitness))
            #print(len(batch))
            snake.score = 0
            snake.respawn(grid, fruit)
            grid.clear(fruit)
            if (netNum == len(batch)-1):
                print("we're here")
        averageFitness/= len(batch)
        fitnessStats.append(averageFitness)
        genStats.append(currentgen)


        plt.scatter(genStats, fitnessStats, marker='o');
        plt.savefig("savedGraphs\GraphGen_"+str(currentgen)+".png")


        fitnessList = []

        print("*****************BEFORE SORTING")
        for i in range(len(batch)):
           print(str(batch[i].fitness) + " before sorting is : " + str(i))
        for i in range(len(batch)):
           print(str(batch[i].fitness))

        # Sort by highest fitness
        fitnessBatch = []
        newBatch = sorted(batch, key=lambda x: x.fitness, reverse = True)

        # Save this batch
        data = np.asarray(newBatch)
        np.save('savedGenerations\Generation_'+str(currentgen)+".npy",data)

        print("******* AFTER SORTING")
        #print(str(max(batch, key=attrgetter('fitness'))))
        for i in range(len(newBatch)):
            fitnessBatch.append(newBatch[i].fitness)
            print(str(newBatch[i].fitness))
        #print(str(max(fitnessBatch)) + " is be fre the max")


        seen = set()
        filteredBatch = []
        for item in newBatch:
            if item.fitness not in seen:
                filteredBatch.append(item)
                seen.add(item.fitness)
                # print(str(item.fitness) + " added")

        #    print(str(batch[i].fitness) + " is : " + str(i))
        print(" ")
        for i in filteredBatch:
            print(i.fitness,end = ",")
        netcounter = 0

        newBatch = filteredBatch[0:select]
        batch = []
        # And repopulate batch with mutations of the top
        for i in range(BATCHSIZE):

                clone = neuralnet.NeuralNet()

                if random.random() < 0.7:
                    clone.syn0 = (filteredBatch[i%(len(filteredBatch))].syn0 + (2 * np.random.random((input_dim + 1, hidden_dim)) - 1) * 0.01)
                    clone.syn1 = (filteredBatch[i%(len(filteredBatch))].syn1 + (2 * np.random.random((hidden_dim + 1, hidden_dim)) - 1) * 0.01)
                    clone.syn2 = (filteredBatch[i%(len(filteredBatch))].syn2 + (2 * np.random.random((hidden_dim + 1, output_dim)) - 1) * 0.01)
                batch.append(clone)



        print(str(len(batch)) + " after")
        newBatch = []
        currentgen+=1

pygame.quit()
