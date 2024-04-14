## Omok
This is an implementation of the  MTCS algorithm for playing the simple board game Omok(Gomoku) for COSE361 assignment.

use distance_policy_value_fn for heuristic function.
you can find it in mtcs.py

### Requirements
you only need:
- Numpy >= 1.11

### Getting Started
To play with provided models, run the following script from the directory:  
```
python human_play.py  
```

Select the first player
Human = 0, AI = 1 : (your input)
Insert the time limit for each move, 0 for infinite
time : (your input)

If you want to play with more powerful AI, Change c_puct and n_playout in human_play.py, line 52
it will cost much more time.
'''
mcts_player = MCTSPlayer(c_puct=5, n_playout=1000)
'''

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/9bee1957-b468-4fe5-b927-4b3ff7675c19/a508b018-4f27-4ba0-8e5f-63e055b3aa4c/Untitled.png)