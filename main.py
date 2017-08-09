# -*- coding: utf-8 -*-

import os
import sys
import datetime

from deepFM import DeepFM
from XP import XP

N_EPOCH = 2


# utils
def trace(text, out_file=sys.stderr):
    print(datetime.datetime.now(), '...', text, file=out_file)


def read_fmdata(fname, batch_size):
    with open(fname, "r") as fp:
        batch = []
        for line in fp:
            line = line.strip()

            label = line.split("|")[0]
            if label in ["1", "1.0"]:
                label = [1]
            elif label in ["0.0", "-1.0", "0", "-1"]:
                label = [0]
            else:
                sys.exit("label tyep invalid!")

            features = [int(x) for x in line.split("|")[1].split(",")]

            batch.append((label, features))

            if (len(batch) == batch_size):
                yield batch
                batch = []

        if batch:
            yield batch


# main
def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    XP.set_library(-1)

    # build model
    model = DeepFM(feature_size=16777216,
                   embed_size=10,
                   hidden_size1=200,
                   hidden_size2=200,
                   hidden_size3=200,
                   nzdim=32)

    model.setup_optimizer("Adam")
    for epoch in range(N_EPOCH):
        trace("start epoch {}/{}".format(epoch+1, N_EPOCH))
        model.opt.new_epoch()
        trained = 0
        for batch_data in read_fmdata(train_file, batch_size=32):
            model.opt.target.zerograds()
            # forward
            loss = model.forward(batch_data, is_training=True)
            # backward
            loss.backward()

            model.opt.update()

            trained += len(batch_data)
            print("\rtrained {}, batch loss = {}".
                  format(trained, loss.data),
                  end="",
                  file=sys.stderr)

        model.save_model("model_epoch{}".format(epoch+1))
        # predict
        with open("predict_epoch{}".format(epoch+1)) as fp:
            for data in read_fmdata(train_file, batch_size=1):
                proba = model.forward(data, is_training=False)
                trace(proba.data)
                print("{0} {1}\n".format(data[0][0], proba.data[0][1]))


if __name__ == '__main__':
    main()
