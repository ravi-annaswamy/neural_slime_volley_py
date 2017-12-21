
"""
Pygame and numpy port, or should I say inspired partial rewrite, 
of selected modules of
David Ha's brilliant Neural Slime Volley js application.

The original application builds a physics model of slimes
playing volley and then endows them with very simple brains
consisting of 12 inputs representing location and speed of ball
and 2 players, 3 outputs representing left, right or jump and
magical 7 hidden state neurons, with a whopping 149 real valued parameters.

Two brilliant implementation ideas make the backend unique, here a
recurrent neural network is simply implemented as a feed forward layer
with hard connection for next step, secondly, David Ha finds weights 
using neural evolution instead of RL.

The results are amazing. Please see his page for details.
http://blog.otoro.net/2015/03/28/neural-slime-volleyball/

This python port 
   1. uncommented
   2. untested, initial sketch, so please do not expect too much.
   2. does not implement neuroevolution so it is not a learning
       ground, just a simulation, I trained a good brain in js and 
       uploaded into a long gene here :)
   3. nowhere close in elegance to otoro artistic beauty!
   
I am neither expert in python nor even newbie in js, simply used chrome js f12 to
study and port this. I learned a lot and I hope python will allow more guys
to appreciate and do a real good port. I am proud of some parts, e.g.
entire Feedforward RNN Brain is 16 lines of python code :)

I think that in many ways, this application is under recognized, (even though well
appreciated for its awesomeness, not many followups even by David Ha!) and 
study and adoption of the core ideas will lead to very good breakthroughs.

12/20/2017
"""

import random
import copy
import numpy as np
import json
#import matplotlib.pyplot as plt # for debug plots during testing
#%matplotlib inline


## VECTOR CLASS FOR PHYSICS

import math
class Vector:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z
    def angle(self):
        assert self.z == 0, "Can't compute the angle for a 3D vector."
        return math.atan2(self.y, self.x)
    def rotate(self, theta):
        self.x = self.x * math.cos(theta) - self.y * math.sin(theta)
        self.y = self.x * math.sin(theta) + self.y * math.cos(theta)
    def angle_between(self, other):
        raise NotImplementedError
    def mag(self):
        return math.sqrt(self.dot(self))
    def set_magnitude_to(self, new_magnitude):
        current_magnitude = self.mag()
        self.x = (self.x / current_magnitude) * new_magnitude
        self.y = (self.y / current_magnitude) * new_magnitude
        self.z = (self.z / current_magnitude) * new_magnitude
    def magnitude_sq(self):
        return self.dot(self)
    def __abs__(self):
        return self.mag()
    def normalize(self):
        self.set_magnitude_to(1)
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
    def __mul__(self, k):
        if isinstance(k, int) or isinstance(k, float):
            return Vector(self.x * k, self.y * k, self.z * k)
        raise TypeError("Can't multiply/divide a vector by a non-numeric.")
    def __rmul__(self, other):
        return self * other
    def __neg__(self):
        return -1 * self
    def __truediv__(self, other):
        return self * (1 / other)
    def cross(self, other):
        return Vector(self.y * other.z - self.z * other.y,
                      self.z * other.x - self.x * other.z,
                      self.x * other.y - self.y * other.x)
    def dot(self, other):
        return sum(sc*oc for sc, oc in zip(self, other))
    def __matmul__(self, other):
        return self.dot(other)
    def __repr__(self):
        return "Vector({:.2f}, {:.2f}, {:.2f})".format(self.x, self.y, self.z)
    __str__ = __repr__

def get_randint(a,b):
    return random.randint(a,b)
def get_randfloat(mini,maxi):
    return random.random()*(maxi-mini)+mini
def createVector(a,b):
    return Vector(a,b)
def get_randcolor():
    return get_randint(127,255),get_randint(127,255),get_randint(127,255)
showArrowKeys = True
ref_w = 24*2
ref_h = ref_w
ref_u = 1.5 # ground height
ref_wallwidth = 1.0 # wall width
ref_wallheight = 3.5
factor = 1
playerSpeedX = 10*1.75
playerSpeedY = 10*1.35
maxBallSpeed = 15*1.5

timeStep = 1/30
theFrameRate = 60*1
nudge = 0.1
friction = 1.0 # 1 means no friction, less means friction
windDrag = 1.0
initDelayFrames = 30*2*1
trainingFrames = 30*20 # assume each match is 7 seconds. (vs 30fps)
theGravity = -9.8*2*1.5
gravity=createVector(0,theGravity)
trainingMode = False
human1 = False # if this is true, then player 1 is controlled by keyboard
human2 = False # same as above
humanHasControlled = False
trainer = None
generationCounter = 0
baseScoreFontSize = 64
trainingVersion = False # this variable is stored on html file pro.html

class Particle:
    def __init__(self, loc=None, v=None, r=None, c=None):
        self.loc=loc or createVector(get_randint(-ref_w*1/4, ref_w*1/4), get_randint(ref_w/4, ref_w*3/4))
        self.prev_loc = copy.deepcopy(self.loc)
        self.v = v or createVector(get_randint(-20,20),get_randint(10,25))
        self.r = r or get_randfloat(0.5,1.5)
        self.c = c or get_randcolor()
        self.ab = copy.deepcopy(self.loc)
    def move(self):
        self.prev_loc.x = self.loc.x
        self.prev_loc.y = self.loc.y
        self.loc+=(self.v * timeStep)
        self.v *= (1-(1-windDrag)*timeStep)
    def apply_acceleration(self,acceleration):
        self.v+=(acceleration*timeStep)
    def check_edges(self):
        if (self.loc.x<=self.r-ref_w/2):
            print('hit left edge')
            print('before',self.loc.x,self.v.x)
            self.v.x *= -friction
            self.loc.x = self.r-ref_w/2+nudge*timeStep
            print('after',self.loc.x,self.v.x)
            return 2
        if (self.loc.x >= (ref_w/2-self.r)):
            print('hit right edge')
            self.v.x *= -friction
            self.loc.x = ref_w/2-self.r-nudge*timeStep
            return 3
        if (self.loc.y<=self.r+ref_u):
            print('ball drop')
            self.v.y *= -friction
            self.loc.y = self.r+ref_u+nudge*timeStep
            if (self.loc.x <= 0):
                return -1
            else:
                return 1
        if (self.loc.y >= (ref_h-self.r)):
            print('hit top')
            self.v.y *= -friction
            self.loc.y = ref_h-self.r-nudge*timeStep
            return 4
        # fence:
        if ((self.loc.x <= (ref_wallwidth/2+self.r)) and (self.prev_loc.x > (ref_wallwidth/2+self.r)) and (self.loc.y <= ref_wallheight)):
            print('hit the fence from right')
            self.v.x *= -friction
            self.loc.x = ref_wallwidth/2+self.r+nudge*timeStep
            return 5
        if ((self.loc.x >= (-ref_wallwidth/2-self.r)) and (self.prev_loc.x < (-ref_wallwidth/2-self.r)) and (self.loc.y <= ref_wallheight)):
            print('hit fence from left')
            self.v.x *= -friction
            self.loc.x = -ref_wallwidth/2-self.r-nudge*timeStep
            return 6
        return 0
    def get_dist_to(self,p):
        '''returns distance squared from p'''
        dy = p.loc.y - self.loc.y
        dx = p.loc.x - self.loc.x
        return (dx*dx+dy*dy)
    def is_colliding(self,p):
        '''returns true if it is colliding w/ p'''
        r = self.r + p.r
        collided = (r*r > self.get_dist_to(p))
        return collided
    def bounce(self,p):
        ab = self.ab
        ab.x=self.loc.x
        ab.y=self.loc.y
        ab-=p.loc
        mag=math.sqrt(ab.x*ab.x+ab.y*ab.y)
        ab/=mag
        ab*=nudge
        while (self.is_colliding(p)):
            self.loc=self.loc+ab
        n = self.loc - p.loc
        mag=math.sqrt(n.x*n.x+n.y*n.y)
        n/=mag
        u = self.v - p.v
        un = n * u.dot(n)*2 # added factor of 2
        u-=un
        self.v = u + p.v
    def limit_speed(self,min_speed, max_speed):
        # untested
        mag2 = self.v.x*self.v.x+self.v.y*self.v.y
        if (mag2 > (max_speed*max_speed) ):
            self.v.normalize()
            self.v*=max_speed
        if (mag2 < (min_speed*min_speed) ):
            self.v.normalize()
            self.v*=min_speed
    """
    def display(self):
      fill(self.c)
      ellipse(toX(self.loc.x), toY(self.loc.y), toP(self.r)*2, toP(self.r)*2)
    """

def pplay_gene_dic():
    gene_json = '{"fitness":1.3846153846153846,"nTrial":0,"gene":{"0":7.5555,"1":4.5121,"2":2.357,"3":0.139,"4":-8.3413,"5":-2.36,"6":-3.3343,"7":0.0262,"8":-7.4142,"9":-8.0999,"10":2.1553,"11":2.4759,"12":1.5587,"13":-0.7062,"14":0.2747,"15":0.1406,"16":0.8988,"17":0.4121,"18":-2.082,"19":1.4061,"20":-12.1837,"21":1.2683,"22":-0.3427,"23":-6.1471,"24":5.064,"25":1.2345,"26":0.3956,"27":-2.5808,"28":0.665,"29":-0.0652,"30":0.1629,"31":-2.3924,"32":-3.9673,"33":-6.1155,"34":5.97,"35":2.9588,"36":6.6727,"37":-2.2779,"38":2.0302,"39":13.094,"40":2.7659,"41":-1.3683,"42":2.5079,"43":-2.6932,"44":-2.0672,"45":-4.2688,"46":-4.9919,"47":-1.1571,"48":-2.0693,"49":2.9565,"50":9.6875,"51":-0.7638,"52":-1.5896,"53":2.4563,"54":-2.5956,"55":-9.8478,"56":-4.9463,"57":-3.4502,"58":-3.0604,"59":-1.158,"60":6.3533,"61":16.0047,"62":1.4911,"63":7.9886,"64":2.3879,"65":-4.5006,"66":-1.8171,"67":0.9859,"68":-2.414,"69":-1.5698,"70":2.5173,"71":-8.6187,"72":-0.3068,"73":-3.6185,"74":-5.202,"75":-0.05,"76":7.2617,"77":-3.1099,"78":0.9881,"79":-0.5022,"80":1.6499,"81":2.1346,"82":2.8479,"83":2.1166,"84":-6.177,"85":0.2584,"86":-3.7623,"87":-4.8107,"88":-9.1331,"89":-2.9681,"90":-7.1177,"91":-1.4894,"92":-1.1885,"93":-4.1906,"94":-5.821,"95":-4.3202,"96":-1.4603,"97":2.3514,"98":-4.8101,"99":3.6935,"100":1.388,"101":3.2504,"102":6.6364,"103":-3.7216,"104":1.6191,"105":6.4388,"106":0.4765,"107":-4.4931,"108":-1.1007,"109":-4.3594,"110":-2.9777,"111":-0.3744,"112":3.5822,"113":3.9402,"114":-9.2382,"115":-4.3392,"116":0.2103,"117":-1.3699,"118":9.2494,"119":10.8483,"120":0.2389,"121":2.6535,"122":-8.2731,"123":-3.5133,"124":-5.0808,"125":3.0846,"126":-0.4851,"127":0.3938,"128":0.2459,"129":-0.3466,"130":-0.1684,"131":-0.7868,"132":-0.6009,"133":2.5491,"134":-3.2234,"135":-3.3352,"136":4.7229,"137":-4.1547,"138":3.6065,"139":-0.1261}}'
    return json.loads(gene_json)['gene']
def tanh(x):
    p=np.exp(x)
    n=np.exp(-x)
    return (p-n)/(p+n)
class Wall:
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.c = (0,200,50)
pruned=[]
class Brain:
    def __init__(self, gene_dict=pplay_gene_dic()): 
        wmat = np.array([gene_dict[str(i)] for i in range(len(gene_dict))])
        weights = wmat[:19*7]
        self.weights = weights.reshape(7,19)
        self.biases = wmat[19*7:]
        self.inputs=np.zeros(19)
        self.outputs=np.zeros(7)
    def set_input(self,own,opponent,ball,direction):
        scaleFactor=10
        #self.inputs[:12]=inp
        self.inputs[0:4]=own.loc.x*direction,own.loc.y,own.v.x*direction,own.v.y
        self.inputs[4:8]=ball.loc.x*direction,ball.loc.y,ball.v.x*direction,ball.v.y
        self.inputs[8:12]=0*opponent.loc.x*direction*-1,0*opponent.loc.y,0*opponent.v.x*direction*-1,0*opponent.v.y
        self.inputs/=scaleFactor
        self.inputs[12:]=self.outputs[-7:]
    def forward(self):
        self.outputs = tanh(np.dot(self.weights,self.inputs)+self.biases)
def test_brain():
    inp1 = np.array([1.2, 0.15, -0, 0, -0, 1.2, 0.12135, 1.2047, 0, 0, 0, 0])
    out1=np.array([0.9999999744281706, 0.9974893427979532, -0.9999994575713337, 0.9999801030130647, 0.9999999898574278, 0.9999998656875623, 0.9996867827035238])
    net = Brain(pplay_gene_dic())
    net.set_input(inp1)
    net.forward()
    assert net.outputs.all()==out1.all()
#test_brain()
class Agent:
    def __init__(self, direction, loc, c):
        self.direction = direction# -1 means left, 1 means right player for symmetry.
        self.loc = loc or createVector(ref_w/4, 1.5)
        self.v = createVector(0, 0)
        self.desiredVelocity = createVector(0, 0)
        self.r = 1.5
        self.c = c
        self.opponent = None
        self.score = 0
        self.emotion = "happy" # hehe...
        self.scoreSize = baseScoreFontSize # font size for score.
        self.action = { # the current set of actions the agent wants to take 
            'forward' : False, # this set of actions can be set either by neural net, or keyboard
            'backward' : False,
            'jump' : False
        }
        self.actionIntensity = [0, 0, 0]
        self.state ={ # complete game state for this agent.  used by neural network.
            'x': 0, # normalized to side, appears different for each agent's perspective
            'y': 0,
            'vx': 0,
            'vy': 0,
            'bx': 0, 
            'by': 0,
            'bvx': 0,
            'bvy': 0
        }
        self.brain = Brain()
    def setOpponent(self, opponent): # sets the opponent into this agent
        self.opponent = opponent
    def setAction(self, forward, backward, jump):
        self.action['forward'] = forward
        self.action['backward'] = backward
        self.action['jump'] = jump
    def setBrainAction(self):
        ''' this def converts the brain's output layer into actions to move forward, backward, or jump'''
        forward = self.brain.outputs[0] > 0.75 # sigmoid decision.
        backward = self.brain.outputs[1] > 0.75 # sigmoid decision.
        jump = self.brain.outputs[2] > 0.75 # sigmoid decision.
        self.setAction(forward, backward, jump)
    def processAction(self): # convert action into real movement
        forward = self.action['forward']
        backward = self.action['backward']
        jump = self.action['jump']
        #print(self.action)
        self.desiredVelocity.x = 0
        self.desiredVelocity.y = 0
        if (forward and not backward):
            self.desiredVelocity.x = -playerSpeedX
        if (backward and not forward):
            self.desiredVelocity.x = playerSpeedX
        if (jump):
            self.desiredVelocity.y = playerSpeedY
    def move(self):
        self.loc+=(self.v*timeStep)
    def getState(self, ball): # returns game state for this agent
        self.state ={ # complete game state for this agent.  used by neural network.
            'x': self.loc.x*self.direction, # normalized to side, appears different for each agent's perspective
            'y': self.loc.y,
            'vx': self.v.x*self.direction,
            'vy': self.v.y,
            'bx': ball.loc.x*self.direction, 
            'by': ball.loc.y,
            'bvx': ball.v.x*self.direction,
            'bvy': ball.v.y
        }
        return self.state

    def update(self):
        self.v+=(gravity*timeStep)
        if (self.loc.y <= ref_u + nudge*timeStep):
            self.v.y = self.desiredVelocity.y
        self.v.x = self.desiredVelocity.x*self.direction
        self.move()
        if (self.loc.y <= ref_u):
            self.loc.y = ref_u
            self.v.y = 0
        # stay in their own half:
        if (self.loc.x*self.direction <= (ref_wallwidth/2+self.r) ):
            self.v.x = 0
            self.loc.x = self.direction*(ref_wallwidth/2+self.r)
        if (self.loc.x*self.direction >= (ref_w/2-self.r) ):
            self.v.x = 0
            self.loc.x = self.direction*(ref_w/2-self.r)

        """
        def display():

          x = self.loc.x
          y = self.loc.y
          r = self.r
          angle = 60
          eyeX = 0
          eyeY = 0

          if (self.dir === 1) angle = 135
          noStroke()
          fill(self.c)
          #ellipse(toX(x), toY(y), toP(r)*2, toP(r)*2)
          arc(toX(x), toY(y), toP(r)*2, toP(r)*2, Math.PI, 2*Math.PI)
          /*
          fill(255)
          rect(toX(x-r), toY(y), 2*r*factor, r*factor)
          */

          # track ball with eyes (replace with observed info later):
          ballX = game.ball.loc.x-(x+(0.6)*r*fastCos(angle))
          ballY = game.ball.loc.y-(y+(0.6)*r*fastSin(angle))
          if (self.emotion === "sad"):
            ballX = -self.dir
            ballY = -3

          dist = Math.sqrt(ballX*ballX+ballY*ballY)
          eyeX = ballX/dist
          eyeY = ballY/dist

          fill(255)
          ellipse(toX(x+(0.6)*r*fastCos(angle)), toY(y+(0.6)*r*fastSin(angle)), toP(r)*0.6, toP(r)*0.6)
          fill(0)
          ellipse(toX(x+(0.6)*r*fastCos(angle)+eyeX*0.15*r), toY(y+(0.6)*r*fastSin(angle)+eyeY*0.15*r), toP(r)*0.2, toP(r)*0.2)
        def drawScore():  
          r = red(self.c)
          g = green(self.c)
          b = blue(self.c)
          size = self.scoreSize
          factor = 0.95
          self.scoreSize = baseScoreFontSize + (self.scoreSize-baseScoreFontSize) * factor

          if (self.score > 0):
            textFont("Courier New")
            textSize(size)
            #stroke(255)
            stroke(r, g, b, 128*(baseScoreFontSize/self.scoreSize))
            fill(r, g, b, 64*(baseScoreFontSize/self.scoreSize))
            textAlign(self.dir === -1? LEFT:RIGHT)
            text(self.score, self.dir === -1? size*3/4 : width-size/4, size/2+height/3)
        """
#game.ball = new Particle(createVector(0, ref_w/4));
#game.ball.r = 0.5;

ground = Wall(0, 0.75, ref_w, ref_u)
fence = Wall(0, 0.75 + ref_wallheight/2, ref_wallwidth, (ref_wallheight - 1.5))
fence.c = (240, 210, 130, 255)
fenceStub = Particle(createVector(0,ref_wallheight), createVector(0,0), \
                    ref_wallwidth/2, (240,210,130))
ball = Particle()
ball.loc.x=0
ball.loc.y=12
#ball.v.x=-4 #-8,12 ends in 650 steps #-10,4 ends in 370 steps
#ball.v.y=22
ball.v.x=random.randint(-10,10)
ball.v.y=random.randint(-10,10)
print('Initial velocity:',ball.v.x,ball.v.y)
ball.r=0.5

agent1 = None
agent1 = agent1 or Agent(-1, createVector(-ref_w/4, 1.5), (240, 75, 0))
print(agent1.v.x,agent1.v.y)
agent2 = None
agent2 = agent2 or Agent(1, createVector(ref_w/4, 1.5), (0, 150, 255))

agent1.setOpponent(agent2) # point agent to the other agent as an opponent.
agent2.setOpponent(agent1)
    

#agent2=Agent(2.0,3.853999999,0,-7.08)
#agent1=Agent(-2.5833333,2.18399999,0,-11.98)

trace_x=[]
trace_y=[]
agent1_x=[]
agent1_y=[]
agent2_x=[]
agent2_y=[]

for i in range(10000):    
    # update internal states
    agent1.getState(ball)
    agent2.getState(ball)

    #push states to brain
    #print(agent1.direction)
    agent1.brain.set_input(agent1, agent2, ball, agent1.direction)
    #print('agent1 inputs',agent1.brain.inputs)
    #print(agent2.direction)
    agent2.brain.set_input(agent2, agent1, ball, agent2.direction)
    #print('agent2 inputs', agent2.brain.inputs)
    
    #make a decision
    agent1.brain.forward()
    #print('agent1 outputs=',agent1.brain.outputs)
    agent2.brain.forward()


    #convert brain's output signals into game actions
    agent1.setBrainAction()
    #print('action=',agent1.action)
    agent2.setBrainAction()


    #process actions
    agent1.processAction()
    agent2.processAction()
    agent1.update()
    agent2.update()    
    
    if i>29:
        ball.apply_acceleration(gravity)
        ball.limit_speed(0,maxBallSpeed)
        ball.move()
        if ball.is_colliding(agent1):
            ball.bounce(agent1)
            print('step',i,'after bounce off agent1:',ball.loc.x,ball.loc.y)
            
        if ball.is_colliding(agent2):
            ball.bounce(agent2)
            print('step',i,'after bounce off agent2:',ball.loc.x,ball.loc.y)

        if ball.is_colliding(fenceStub):
            ball.bounce(fenceStub)
            print('step',i,'after bounce off fenceStub:',ball.loc.x,ball.loc.y)

            
        result = ball.check_edges()
        if result != 0:
            print('Hit at step ',i,result)
            print(i,str(ball.loc.x)[:7],str(ball.loc.y)[:7],str(ball.v.x)[:5],str(ball.v.y)[:5])
            
            if result in [1,-1]:
                print('Game Ends at step:',i, result)
                break
    #print(i,str(ball.loc.x),str(ball.loc.y),str(ball.v.x),str(ball.v.y))
    trace_x.append(ball.loc.x)
    trace_y.append(ball.loc.y)
    agent1_x.append(agent1.loc.x)
    agent1_y.append(agent1.loc.y)
    agent2_x.append(agent2.loc.x)
    agent2_y.append(agent2.loc.y)
            

import pygame
import time
screen = pygame.display.set_mode((500, 500))
pygame.display.update()
background_colour = (255,255,255)
while True:
    for i in range(len(trace_x)):
        #print(i, 250+int(trace_x[i]*10), 500-int(trace_y[i]*10))
        screen.fill(background_colour)
        pygame.draw.rect(screen, (0,0,0), ((245,470), (10,30)))
        pygame.draw.circle(screen, (255,0,255), (250+int(trace_x[i]*10), 500-int(trace_y[i]*10)), 10, 0)
        pygame.draw.ellipse(screen,(0,0,255), ((250+int(agent1_x[i]*10)-15, 500-int(agent1_y[i]*10)-15), (30,30)), 0)
        pygame.draw.ellipse(screen,(0,255,0), ((250+int(agent2_x[i]*10)-15, 500-int(agent2_y[i]*10)-15), (30,30)), 0)
        #for k in range(0,500,125):
        #    pygame.draw.line(screen,(50,0,0),(k,0),(k,500))
        pygame.display.flip()
        time.sleep(0.03)
        
        #You need to regularly make a call to one of four functions in the pygame.event module in order for 
        #pygame to internally interact with your OS. Otherwise the OS will think your game has crashed. 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit() 
        if i==len(trace_x)-1:
            break
    if i==len(trace_x)-1:
        break
#print(sorted(pruned))
