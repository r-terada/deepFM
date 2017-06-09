# -*- coding: utf-8 -*-
import os
import sys

from XP import XP

from chainer import functions as F
from chainer import links as L
from chainer import Chain
from chainer import serializers
from chainer import optimizers, optimizer


class DeepFM(Chain):
    def __init__(self,
                 feature_size=16777216,
                 embed_size=5,
                 hidden_size1=200,
                 hidden_size2=200,
                 hidden_size3=200,
                 nzdim=32):
        super(DeepFM, self).__init__(
            # Shared feature embedding
            embed=L.EmbedID(feature_size, embed_size, ignore_label=-1),
            # FM Component
            L1=L.EmbedID(feature_size, 1),
            # Deep Component (to capture higher order intaractions)
            L2=L.Linear(nzdim * embed_size, hidden_size1),
            L3=L.Linear(hidden_size1, hidden_size2),
            L4=L.Linear(hidden_size2, hidden_size3),
            L5=L.Linear(hidden_size3, 1)
        )

        self.feature_size = feature_size
        self.embed_size = embed_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.nzdim = nzdim

    def setup_optimizer(self, optimizer_name, gradient_clipping=3,
                        weight_decay=0.00001, **kwargs):
        # set optimizer
        if optimizer_name == "Adam":
            self.opt = optimizers.Adam(**kwargs)
        elif optimizer_name == "AdaDelta":
            self.opt = optimizers.AdaDelta(**kwargs)
        elif optimizer_name == "AdaGrad":
            self.opt = optimizers.AdaGrad(**kwargs)
        elif optimizer_name == "RMSprop":
            self.opt = optimizers.RMSprop(**kwargs)
        elif optimizer_name == "RMSpropGraves":
            self.opt = optimizers.RMSpropGraves(**kwargs)
        elif optimizer_name == "SGD":
            self.opt = optimizers.SGD(**kwargs)
        elif optimizer_name == "MomentumSGD":
            self.opt = optimizers.MomentumSGD(**kwargs)

        # self.opt.use_cleargrads()
        self.opt.setup(self)
        self.opt.add_hook(optimizer.GradientClipping(gradient_clipping))
        self.opt.add_hook(optimizer.WeightDecay(weight_decay))

        self.opt_params = {
            "optimizer_name": optimizer_name,
            "gradient_clipping": gradient_clipping,
            "weight_decay": weight_decay
        }

    def forward(self, data, is_training=True):
        """
        data: [([label], [features]), ([label], [features]), ...)]
        """

        x_raw = [x[1] for x in data]
        batch_size = len(x_raw)
        # embed sparse featuer vector to dense vector
        x_sparse = XP.iarray(x_raw)
        x_sparse = F.reshape(x_sparse, [batch_size * self.nzdim])
        embeddings = self.embed(x_sparse)

        # FM Component
        # 1st order
        first_order = F.reshape(
            F.sum(F.reshape(self.L1(x_sparse), (batch_size, self.nzdim)), 1)
            , (batch_size, 1)
            )

        # 2nd order
        embeddings = F.reshape(
            embeddings, (batch_size, self.nzdim * self.embed_size)
        )

        second_order = XP.fzeros((batch_size, 1))
        for i in range(self.nzdim-1):
            for j in range(1, self.nzdim-i):
                former = embeddings[:, i*self.embed_size:(i+1)*self.embed_size]
                later = embeddings[:, (i+j)*self.embed_size:(i+j+1)*self.embed_size]

                second_order += F.reshape(F.batch_matmul(former, later, transa=True), (batch_size, 1))

        y_fm = first_order + second_order
        # Deep Component
        embeddings = F.reshape(
            embeddings, (batch_size, self.nzdim * self.embed_size)
        )
        h = F.dropout(
                F.relu(self.L2(embeddings)),
                ratio=0.9, train=is_training
            )
        h = F.dropout(
                F.relu(self.L3(h)),
                ratio=0.9, train=is_training
            )
        h = F.dropout(
                F.relu(self.L4(h)),
                ratio=0.9, train=is_training
            )
        y_deep = self.L5(h)

        y = y_fm + y_deep

        if is_training:
            t_raw = [t[0] for t in data]
            t = XP.iarray(t_raw)
            return F.sigmoid_cross_entropy(y, t)
        else:
            return F.sigmoid(y)

    def _save_params(self, filename):
        params = {"feature_size": self.feature_size,
                  "embed_size": self.embed_size,
                  "hidden_size1": self.hidden_size1,
                  "hidden_size2": self.hidden_size2,
                  "hidden_size3": self.hidden_size3,
                  "nzdim": self.nzdim}
        with open(filename, 'w') as fp:
            json.dump(params, fp)

    def _save_weights(self, filename):
        serializers.save_hdf5(filename, self)

    def save_model(self, filename):
        self._save_params(filename+".params")
        self._save_weights(filename+".weights")

    def load_model(self, filename):
        """
        Load parameters and weights from file
        """
        if "/" in filename:
            basename = "/".join(filename.split("/")[:-1]) + "/" + filename.split("/")[-1].split(".")[0]
        else:
            basename = filename.split(".")[0]
        serializers.load_hdf5(filename + '.weights', self)
