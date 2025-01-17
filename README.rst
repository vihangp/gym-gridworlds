OpenAI gym Gridworlds
=====================

Implementation of three gridworlds environments
from book `Reinforcement Learning: An Introduction
<http://incompleteideas.net/book/the-book-2nd.html>`_
compatible with `OpenAI gym <https://github.com/openai/gym>`_.

Usage
-----

.. code::

        $ import gym
        $ import gym_gridworlds
        $ env = gym.make('Gridworld-v0')  # substitute environment's name

``Gridworld-v0``
----------------

Gridworld is simple 4 times 4 gridworld from example 4.1 in the [book].
There are four action in each state (up, down, right, left)
which deterministically cause the corresponding state transitions
but actions that would take an agent of the grid leave a state unchanged.
The reward is -1 for all tranistion until the terminal state is reached.
The terminal state is in top left and bottom right coners.

``WindyGridworld-v0``
---------------------

Windy gridworld is from example 6.5 in the book_.
Windy gridworld is a standard gridworld as described above
but there is a crosswind upward through the middle of the grid.
Action are standard but in the middle region the resultant states are
shifted upward by a wind which strength varies between columns.

.. _book: http://incompleteideas.net/book/the-book-2nd.html

``Cliff-v0``
------------

Cliff walking is a gridworld example 6.6 from the book_.
Again reward is -1 on all transition except those into region
that is cliff.
Stepping into this region incurs a reward of -100
and sends the agent instantly back to the start.

``DistractingCliff-v0``
------------

This is a modified version of the Cliff-v0.
Here the reward is 1 on all transition except those into region
that is cliff.
Stepping into this region incurs a reward of "r", where r is less
than the return accumulated if the agent goes through all the states. Also,
reaching the cliff, ends the episode.

``ExplorationGrid-v0``
------------

This is a modified version of the Cliff-v0.
Here the reward is 1 on all transition and there are few transitions, where the
reward is 5. The agent cannot get the reward for the same state again within the same reward, once it has
entered some state. The episode ends, once it goes to the bottom right corner.