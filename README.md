# image
```
docker build -t rl .
```

# gather the data of plays
```
docker run -d --gpus=all -v $PWD:/app rl python main2.py --model-ver=? --moves-cutoff=<int> --n-rollout=<int> --n-epochs=<int> --save-all=<bool>
```

where, the `$PWD` is the current working directory of the full code. The beton files are written
in this folder.

The `moves-cutoff` is the maximal number of steps for WHITE and BLACK togather to play. Exceeding the number is an unfinished game.

By default, only the finished games are saved as a training data. `--save-all` will save all plays.

# train with the gathered data
Place beton files in `v<model-ver>/dataset/`.
```
docker run -d --gpus=all -v $PWD:/app rl python train.py --model-ver=<int>
```

The trained model is in the `v<model-ver>/checkpoint/`
