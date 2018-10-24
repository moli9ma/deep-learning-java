package net.moli9ma.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SimpleNet {

    INDArray weight;

    public SimpleNet(INDArray weight) {
        this.weight = weight;
    }

    /**
     * 予測
     *
     * @param input
     * @return
     */
    public INDArray predict(INDArray input) {
        return input.mmul(this.weight);
    }

    /**
     * 損失関数
     *
     * @param input
     * @param t
     * @return
     */
    public double loss(INDArray input, INDArray t) {

        // 入力との重みを計算する
        INDArray z = this.predict(input);

        // 重みづけ処理した結果に対して、確率分布を求める
        INDArray y = NdUtil.Softmax(z);

        // 交差エントロピー誤差を求める
        double loss = NdUtil.CrossEntropyError(y, t);
        return loss;
    }
}
