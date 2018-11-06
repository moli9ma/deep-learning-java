package net.moli9ma.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.Function;

public class TwoLayerNet {

    public TwolayerNetParameter parameter;

    public TwoLayerNet(INDArray weight1, INDArray bias1, INDArray weight2, INDArray bias2) {
        parameter = new TwolayerNetParameter(weight1, bias1, weight2, bias2);
    }

    public TwoLayerNet(int inputSize, int hiddenSize, int outputSize, double weightInitStd) {
        INDArray weight1 = Nd4j.rand(inputSize, hiddenSize).mul(weightInitStd);
        INDArray weight2 = Nd4j.rand(hiddenSize, outputSize).mul(weightInitStd);

        INDArray bias1 = Nd4j.zeros(hiddenSize);
        INDArray bias2 = Nd4j.zeros(outputSize);
        parameter = new TwolayerNetParameter(weight1, bias1, weight2, bias2);
    }

    /**
     * 予測
     *
     * @param input
     * @return
     */
    public INDArray predict(INDArray input) {

        INDArray result1 = input.mmul(this.parameter.weight1).addRowVector(this.parameter.bias1);
        INDArray sigResult1 = NdUtil.Sigmoid(result1);

        INDArray result2 = sigResult1.mmul(this.parameter.weight2).addRowVector(this.parameter.bias2);
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
     * 勾配を計算
     *
     * @param x 入力データ
     * @param t 教師データ
     * @return
     */
    public TwolayerNetParameter numericalGradient(INDArray x, INDArray t) {
        Function<INDArray, Double> lossFunc = w -> this.loss(x, t);
        INDArray weight1 = NdUtil.NumericalGradient(lossFunc, this.parameter.weight1);
        INDArray bias1 = NdUtil.NumericalGradient(lossFunc, this.parameter.bias1);
        INDArray weight2 = NdUtil.NumericalGradient(lossFunc, this.parameter.weight2);
        INDArray bias2 = NdUtil.NumericalGradient(lossFunc, this.parameter.bias2);
        return new TwolayerNetParameter(weight1, bias1, weight2, bias2);
    }

}
