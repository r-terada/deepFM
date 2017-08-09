# chainer implementation of DeepFM

- [DeepFM](https://arxiv.org/abs/1703.04247)

```
pip install chainer==1.17.0
python main.py train.txt test.txt
```

- train.txt and test.txt should be following

```
# label|feature1,feature2,feature3,feature4,...
# positive labels are (1, 1.0)
# negative labels are (-1, -1.0, 0, 0.0)

1.0|12345,1280803,123,4,...
```