# RL-Environment

A Reinforcement Learning environment represents a problem, or a game, that can
be solved by agents via Reinforcement Learning. Many environments are available,
but it is quite interesting to create our own game and see the agent play it.
In this project a personalized game is created and Q-Learning is used to teach an
agent how to play it. Of course, many improvements are possible, but not always
with the limited performance of the average personal computer.

The game has three main components, the player, the enemy, and the safe zone. The idea behind it is
that the player is outside its base, the safe zone, and has to go back without being captured by the
enemy and before the enemy realizes that the base is empty, because otherwise it will conquer it. The
rules are therefore quite simple: the player must reach the safe zone without being touched by the
enemy and before the latter reaches it.
