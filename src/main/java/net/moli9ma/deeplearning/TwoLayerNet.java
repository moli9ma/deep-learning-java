package net.moli9ma.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.Function;

public class TwoLayerNet {


    INDArray weight1;
    INDArray bias1;

    INDArray weight2;
    INDArray bias2;

    /**
     * @param weight1 重み1
     * @param bias1   バイアス1
     * @param weight2 重み2
     * @param bias2   バイアス2
     */
    public TwoLayerNet(INDArray weight1, INDArray bias1, INDArray weight2, INDArray bias2) {
        this.weight1 = weight1;
        this.bias1 = bias1;
        this.weight2 = weight2;
        this.bias2 = bias2;
    }

    /**
     * 予測
     *
     * @param input
     * @return
     */
    public INDArray predict(INDArray input) {

        INDArray result1 = input.mmul(weight1).addRowVector(bias1);
        INDArray sigResult1 = NdUtil.Sigmoid(result1);

        INDArray result2 = sigResult1.mmul(weight2).addRowVector(bias2);
        INDArray softmaxResult2 = NdUtil.Softmax(result2);

        return softmaxResult2;
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

        // 交差エントロピー誤差を求める
        double loss = NdUtil.CrossEntropyError(z, t);
        return loss;
    }

    /**
     * 認識精度を返却します。
     *
     * @param x
     * @param t
     * @return
     */
    public double accuracy(INDArray x, INDArray t) {

        INDArray predicted = this.predict(x);
        predicted = Nd4j.argMax(predicted, 1);
        INDArray y = Nd4j.argMax(t, 1);

        double correctNumber = predicted.eq(y).sumNumber().doubleValue();
        long size = t.size(0);

        return correctNumber / size;
    }

    /**
     *
     *
     * @param x 入力データ
     * @param t 教師データ
     * @return
     */
    public void numericalGradient(INDArray x, INDArray t) {
        Function<INDArray, Double> lossFunc = w -> this.loss(x, t);
        this.weight1 = NdUtil.NumericalGradient(lossFunc, this.weight1);
        this.bias1 = NdUtil.NumericalGradient(lossFunc, this.bias1);
        this.weight2 = NdUtil.NumericalGradient(lossFunc, this.weight2);
        this.bias2 = NdUtil.NumericalGradient(lossFunc, this.bias2);
    }

}
