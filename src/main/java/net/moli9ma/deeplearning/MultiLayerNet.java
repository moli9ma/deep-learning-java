package net.moli9ma.deeplearning;

import net.moli9ma.deeplearning.layer.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.function.Function;

public class MultiLayerNet {

    long inputSize;
    long outputSize;
    List<Long> hiddenSizeList;
    ActivationType activationType;
    WeightInitializeType weightInitializeType;
    double weightInitStd;
    double weightDecayLambda;

    HashMap<String, Layer> layers;
    HashMap<String, INDArray> params;

    LastLayer lastLayer;

    /**
     * Constructor
     * @param inputSize 入力サイズ（MNISTの場合は784）
     * @param outputSize  出力サイズ（MNISTの場合は10
     * @param hiddenSizeList  隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
     * @param activationType  ReLU or Sigmoid
     * @param weightInitStd 重みの標準偏差を指定（e.g. 0.01）
     *         'relu'または'he'を指定した場合は「Heの初期値」を設定
     *         'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
     * @param weightDecayLambda Weight Decay（L2ノルム）の強さ
     */
    public MultiLayerNet(long inputSize, long outputSize, List<Long> hiddenSizeList, ActivationType activationType, WeightInitializeType weightInitializeType ,double weightInitStd, double weightDecayLambda) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.hiddenSizeList = hiddenSizeList;
        this.activationType = activationType;
        this.weightInitStd = weightInitStd;
        this.weightDecayLambda = weightDecayLambda;


        this.initWeight(weightInitStd);

        for (int i = 0; i < hiddenSizeList.size(); i++) {
            this.layers.put("Affine" + i, new AffineLayer(this.params.get("W" + i), this.params.get("b" + i)));
            this.layers.put("Activation_function" + i, toActivationFunc(activationType));
        }

        long index = hiddenSizeList.size();
        this.layers.put("Affine" + index, new AffineLayer(this.params.get("W" + index), this.params.get("b" + index)));
        this.lastLayer = new SoftmaxWithLossLayer();
    }

    /**
     * 重みを初期化します。
     * @param weightInitStd
     */
    private void initWeight(double weightInitStd) {

        List<Long> allSizeList = new ArrayList<>();
        allSizeList.add(this.inputSize);
        allSizeList.addAll(this.hiddenSizeList);
        allSizeList.add(this.outputSize);

        for (int i = 0; i < allSizeList.size(); i++) {
            double scale = weightInitStd;
            switch (weightInitializeType) {
                case ReLU:
                case He:
                    scale = Math.sqrt(2.0 / allSizeList.get(i));
                case Sigmoid:
                case Xavier:
                    scale = Math.sqrt(1.0 / allSizeList.get(i));
            }

            this.params.put("W" + i, Nd4j.randn(allSizeList.get(i), allSizeList.get(i+1)).mul(scale));
            this.params.put("b" + i, Nd4j.zeros(allSizeList.get(i+1)));
        }
    }

    /**
     * 予測
     * @param x
     * @return
     */
    public INDArray predict (INDArray x) {
        for (Layer layer : this.layers.values()) {
                x = layer.forward(x);
        }
        return x;
    }

    /**
     * 損失
     *
     * @param x 入力データ
     * @param t 教師データ
     * @return
     */
    public double loss (INDArray x, INDArray t) {
        INDArray y = this.predict(x);

        double weightDecay = 0;
        for (int i = 0; i < this.hiddenSizeList.size() + 1; i++) {
            INDArray w = this.params.get("W" + i);
            weightDecay += 0.5 * this.weightDecayLambda * Transforms.pow(w, 2).sumNumber().doubleValue();
        }
        return this.lastLayer.forward(y, t) + weightDecay;
    }

    /**
     * 認識精度
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
     * 勾配を求める
     *
     * @param x 入力データ
     * @param t 教師データ
     *
     * @return
     *     各層の勾配を持ったディクショナリ変数
     *     grads['W1']、grads['W2']、...は各層の重み
     *     grads['b1']、grads['b2']、...は各層のバイアス
     */
    public HashMap<String, INDArray> numericalGradient(INDArray x, INDArray t) {

        Function<INDArray, Double> lossFunc = w -> this.loss(x, t);
        HashMap<String, INDArray> grads = new HashMap<>();
        for (int i = 0; i < this.hiddenSizeList.size() + 1; i++) {
            grads.put("W" + i, NdUtil.NumericalGradient(lossFunc, this.params.get("W" + i)));
            grads.put("b" + i, NdUtil.NumericalGradient(lossFunc, this.params.get("b" + i)));
        }
        return grads;
    }

    /**
     * 勾配を求める(誤差逆伝搬法)
     * @param x
     * @param t
     * @return
     *     各層の勾配を持ったディクショナリ変数
     *     grads['W1']、grads['W2']、...は各層の重み
     *     grads['b1']、grads['b2']、...は各層のバイアス
     */
    public HashMap<String, INDArray> gradient(INDArray x, INDArray t) {

        // forward
        this.loss(x, t);

        // backward for lastLayer
        INDArray b = this.lastLayer.backward();


        // backward for layers
        List<String> reverseOrderedKeys = new ArrayList<>(this.layers.keySet());
        Collections.reverse(reverseOrderedKeys);
        for (String key : reverseOrderedKeys) {
            b = this.layers.get(key).backward(b);
        }

        HashMap<String, INDArray> grads = new HashMap<>();
        for (int i = 0; i < this.hiddenSizeList.size() + 1; i++) {
            AffineLayer affineLayer = (AffineLayer)this.layers.get("Affine" + i);
            grads.put("W" + i, affineLayer.dWeight.add(this.weightDecayLambda).mul(affineLayer.weight));
            grads.put("b" + i, affineLayer.dBias);
        }
        return grads;
    }

    /**
     * 重み初期化タイプ
     */
    public enum WeightInitializeType {
        ReLU,
        Sigmoid,
        He,
        Xavier
    }

    /**
     * アクティベーションタイプ
     */
    public enum ActivationType {
        ReLU,
        Sigmoid,
    }

    /**
     * ActivationTypeに対応する関数を返却します。
     * @param activationType
     * @return
     */
    private Layer toActivationFunc(ActivationType activationType) {
        switch (activationType) {
            case ReLU:
                return new ReLULayer();

            case Sigmoid:
                return new SigmoidLayer();

        }
        return new SigmoidLayer();
    }


}



