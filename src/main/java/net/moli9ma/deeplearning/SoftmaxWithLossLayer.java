package net.moli9ma.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;

public class SoftmaxWithLossLayer {

    /**
     * 損失
     */
    double loss;

    /**
     * softmaxの出力
     */
    INDArray y;

    /**
     * 教師データ
     */
    INDArray t;


    public double forward(INDArray x, INDArray t) {
        this.y = NdUtil.Softmax(x);
        this.t = t;
        this.loss = NdUtil.CrossEntropyError(this.y, this.t);
        return loss;
    }

    public INDArray backward() {
        long batch_size = this.t.size(0);
        return this.y.sub(this.t).div(batch_size);
    }

}
