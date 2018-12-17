package net.moli9ma.deeplearning.layer;

import net.moli9ma.deeplearning.NdUtil;
import net.moli9ma.deeplearning.layer.LastLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class SoftmaxWithLossLayer implements LastLayer {

    /**
     * 損失
     */
    public double loss;

    /**
     * softmaxの出力
     */
    public INDArray y;

    /**
     * 教師データ
     */
    public INDArray t;


    @Override
    public double forward(INDArray x, INDArray t) {
        this.y = Transforms.softmax(x);
        this.t = t;
        this.loss = NdUtil.CrossEntropyError(this.y, this.t);
        return loss;
    }

    @Override
    public INDArray backward() {
        long batch_size = this.t.size(0);
        return this.y.sub(this.t).div(batch_size);
    }
}

/*
class SoftmaxWithLoss:
        def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

        def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

        def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
        dx = (self.y - self.t) / batch_size
        else:
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx = dx / batch_size

        return dx
*/
