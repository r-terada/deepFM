# DeepFM 実装

- [DeepFM](https://arxiv.org/abs/1703.04247) を実装しました.
- training 遅すぎ (350 examples/min くらい) なので全然実験終わりませんでした.

```
pip install chainer==1.17.0
python main.py validation/train.txt validation/test.txt
```