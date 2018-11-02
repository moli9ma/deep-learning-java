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
