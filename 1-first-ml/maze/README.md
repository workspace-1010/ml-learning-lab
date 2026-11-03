# Project 2 — Teach an agent to solve a maze

In this project, you will build a tiny game environment and train an agent to find the goal using **Q-learning**. The agent starts without knowing the route. It tries actions, receives rewards or penalties, and gradually learns which actions are useful.

This is reinforcement learning: the agent learns from interaction rather than from a dataset of correct answers.

## What you will build

- A maze represented by a grid of walls and open spaces.
- An agent that can move up, down, left, or right.
- A reward system: reaching the goal is good, each step has a small cost, and walking into a wall is penalized.
- A Q-table that stores how useful each action appears to be from each position.
- A training loop that balances exploration (trying moves) with exploitation (using what it has learned).
- A final display of the route the trained agent chooses.

## Libraries and tools

| Library/tool | Built into Python? | Why we use it |
|---|---|---|
| Python 3.10+ | No; install Python itself | Runs the project |
| `random` | Yes — Python standard library | Chooses exploratory actions |
| `argparse` | Yes — Python standard library | Lets you change the number of training episodes from the command line |
| `collections` | Yes — Python standard library | Stores the Q-values by state and action |
| `dataclasses` | Yes — Python standard library | Defines the maze environment cleanly |

**No NumPy, scikit-learn, PyTorch, game engine, or external dataset is required.** This project is intentionally built with Python's standard library so you can focus on the learning algorithm.

## Setup and run

Follow the [main setup guide](../../docs/setup-guide.md) to set up Python and the repository. No additional package installation is needed for this project.

Run from the repository root:

```bash
python 1-first-ml/maze/01_train_maze_agent.py
```

To train for more or fewer episodes:

```bash
python 1-first-ml/maze/01_train_maze_agent.py --episodes 10000
```

The script prints the maze, training progress, and the route chosen by the trained agent. It also reports how many steps the final route took. Training includes randomness, so the route and progress figures can vary between runs.

## How Q-learning works

For a position (state) and an action, the Q-table stores an estimate of how useful that action is. After the agent moves, it updates that estimate using the reward and its estimate of future reward:

`Q(s, a) ← Q(s, a) + α [r + γ maxₐ′ Q(s′, a′) − Q(s, a)]`

- **State `s`:** the agent's current grid position.
- **Action `a`:** up, down, left, or right.
- **Reward `r`:** the immediate feedback for the move.
- **Next state `s′`:** the position after the move.
- **Learning rate `α`:** how much a new experience changes the current estimate.
- **Discount factor `γ`:** how much future rewards matter.
- **Exploration rate `ε`:** the chance of trying a random move instead of the current best-known move.

The agent does **not** receive a map showing the correct path as a set of answers. It discovers useful moves by trying actions and observing the reward. The environment's grid defines where walls and the goal are, as it would in a game.

## Experiments to try

1. Run with 100 episodes, then 1,000, then 10,000. Compare the route.
2. Change the exploration rate or how quickly it decays. Does the agent explore enough?
3. Make each step more expensive or increase the goal reward. How does that affect the learned policy?
4. Draw a different maze and see whether the agent can learn that route too.

## Troubleshooting

- Run the command from the repository root so the path is correct.
- If the agent occasionally fails to find a route after a small number of episodes, increase `--episodes`.
- The route can vary because exploration uses randomness; that is expected.
- This is a small educational tabular Q-learning agent, not a neural-network-based reinforcement-learning system.

## How this follows the MNIST project

MNIST teaches supervised learning: the model learns from images paired with correct labels. The maze teaches reinforcement learning: the agent learns from actions, rewards, and delayed consequences. Together they demonstrate two different ways a machine-learning system can learn.
