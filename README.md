# image
```
docker build -t rl .
```

# gather the data of plays
Update the `MOVES_CUTOFF`, `N_ROLLOUT`, `MODEL_VER` (identitying a trained model for one/both players), and set the `player1` and `player2` in the main routine.

```
docker run -d --gpus=all -v $PWD:/app rl python main.py
```

where, the `$PWD` is the current working directory of the full code. The beton files are written
in this folder.

# train with the gathered data
Update the `MODEL_VER` to some folder (any str, identitying the version of the model), and place
beton files in `{MODEL_VER}/dataset/`.
```
docker run -d --gpus=all -v $PWD:/app rl python train.py
```

The trained model is in the `{MODEL_VER}/checkpoint/`
