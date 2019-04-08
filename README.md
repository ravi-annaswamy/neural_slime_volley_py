# neural_slime_volley_py

## Partial port of otoro excellent javascript Neural Slime Volleyball app

Pygame and numpy port, or should I say inspired partial rewrite, of selected modules of
David Ha's brilliant Neural Slime Volley js application.

The original application builds a physics model of slimes
playing volley and then endows them with very simple brains
consisting of 12 inputs.

These 12 represent:
 - x,y location and vx,vy speed of ball
 - likewise the location and speed of the 2 players
 
The job of the network brain is to decide 3 outputs representing 
it possible actions: move left, right or jump.

With 'magical' just 7 hidden state neurons, with a 'whopping' 149 real valued parameters,
the network weights lead to high quality play!!!

Two brilliant implementation ideas make the backend unique, here a
recurrent neural network is simply implemented as a feed forward layer
with hard connection for next step, secondly, David Ha finds weights 
using neural evolution instead of RL.<br>


The results are amazing. Please see his [page for details.]
(http://blog.otoro.net/2015/03/28/neural-slime-volleyball/)

This python port is <br>
   1. uncommented
   2. untested, initial sketch, so please do not expect too much.
   3. does not implement neuroevolution so it is not a learning
       ground, just a simulation, I trained a good brain in js and 
       uploaded into a long gene here :)
   3. nowhere close in elegance to otoro artistic beauty!


I am neither expert in python nor even newbie in js, simply used chrome js f12 to
study and port this. 

I learned a lot and I hope python will allow more guys
to appreciate and do a real good port. 

I am proud of some parts, e.g.
entire Feedforward RNN Brain is 16 lines of python code :)


I think that in many ways, this application is under recognized, (even though well
appreciated for its awesomeness, not many followups even by David Ha!) and 
study and adoption of the core ideas will lead to very good breakthroughs.

Ravi
12/20/2017

UPDATE on 4/8/2019:

When I first saw this learning through self play, I had noted that this should be applied
for Game playing. Later, it was a pleasure to see that Alphago Zero worked better than
initialized AlphaGo.

This is because an agent playing itself, creates optimal learning path for itself,
since its opponent uses concepts and techniques in the same 'level' of understanding as 
itself and hence there is lower likelihood of overfitting!

