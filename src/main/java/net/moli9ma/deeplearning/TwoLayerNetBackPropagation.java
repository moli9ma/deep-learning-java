package net.moli9ma.deeplearning;

import javafx.scene.transform.Affine;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.function.Function;

public class TwoLayerNetBackPropagation {

    TwolayerNetParameter parameter;
    LinkedHashMap<String, Layer> layers;
    LastLayer lastLayer;


    public TwoLayerNetBackPropagation(int inputSize, int hiddenSize, int outputSize, double weightInitStd) {

        Random r = new DefaultRandom(0);
        INDArray weight1 = r.nextGaussian(new int[] {inputSize, hiddenSize}).mul(weightInitStd);
        INDArray weight2 = r.nextGaussian(new int[] {hiddenSize, outputSize}).mul(weightInitStd);

/*
        INDArray weight1 = Nd4j.rand(inputSize, hiddenSize).mul(weightInitStd);
        INDArray weight2 = Nd4j.rand(hiddenSize, outputSize).mul(weightInitStd);
*/

        INDArray bias1 = Nd4j.zeros(hiddenSize);
        INDArray bias2 = Nd4j.zeros(outputSize);

        LinkedHashMap<String, Layer> layers = new LinkedHashMap<>();
        layers.put("Affine1", new AffineLayer(weight1, bias1));
        layers.put("ReLU1", new ReLULayer());
        layers.put("Affine2", new AffineLayer(weight2, bias2));

        this.parameter = new TwolayerNetParameter(weight1, bias1, weight2, bias2);
        this.lastLayer = new SoftmaxWithLossLayer();
        this.layers = layers;
    }


    /**
     * 予測
     *
     * @param input
     * @return
     */
    public INDArray predict(INDArray input) {
        for (Layer layer : this.layers.values()){
            input = layer.forward(input);
        }
        return input;
    }

    /**
     * 損失関数
     *
     * @param input
     * @param t
     * @return
     */
    public double loss(INDArray input, INDArray t) {
        INDArray z = this.predict(input);
        return this.lastLayer.forward(z, t);
    }

    /**
     * 認識精度を返却します。
     *
     * @param x
     * @param t
     * @return
     */
    public double accuracy(INDArray x, INDArray t) {
        INDArray y = predict(x);
        y = Nd4j.argMax(y, 1);
        if (t.size(0) != 1)
            t = Nd4j.argMax(t, 1);
        double accuracy = y.eq(t.broadcast(y.shape())).sumNumber().doubleValue() / x.size(0);
        return accuracy;
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

    /**
     * 誤差逆伝搬法によって重みパラメータの勾配を求める
     *
     * @param x
     * @param t
     * @return
     */
    public TwolayerNetParameter gradient(INDArray x, INDArray t) {
        this.loss(x, t);
        INDArray b = this.lastLayer.backward();

        List<String> reverseOrderedKeys = new ArrayList<>(this.layers.keySet());
        Collections.reverse(reverseOrderedKeys);

        for (String key : reverseOrderedKeys) {
            b = this.layers.get(key).backward(b);
        }

        AffineLayer affineLayer1 = (AffineLayer) this.layers.get("Affine1");
        INDArray weight1 = affineLayer1.dWeight;
        INDArray bias1 = affineLayer1.dBias;

        AffineLayer affineLayer2 = (AffineLayer) this.layers.get("Affine2");
        INDArray weight2 = affineLayer2.dWeight;
        INDArray bias2 = affineLayer2.dBias;

        return new TwolayerNetParameter(weight1, bias1, weight2, bias2);
    }
}
