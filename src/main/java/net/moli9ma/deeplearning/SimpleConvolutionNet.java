package net.moli9ma.deeplearning;

import net.moli9ma.deeplearning.layer.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.function.Function;

/**
 * 単純な畳み込みネットワーク
 * conv - relu - pool - affine - relu - affine - softmax
 */
public class SimpleConvolutionNet implements Network {

    /**
     * 入力サイズ(幅 MNISTの場合は28）
     */
    private final int inputWidth;

    /**
     * 入力サイズ(高さ MNISTの場合は28)
     */
    private final int inputHeight;

    /**
     * 隠れ層のニューロンの数のサイズ
     */
    private final int hiddenSize;

    /**
     * 出力サイズ（MNISTの場合は10）
     */
    private final int outputSize;

    /**
     * 'relu' or 'sigmoid'
     */
    private final ActivationType activationType;

    /**
     * 重みの標準偏差を指定（e.g. 0.01）
     * 'relu'または'he'を指定した場合は「Heの初期値」を設定
     * 'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
     */
    private final WeightInitializeType weightInitializeType;

    /**
     * カーネル数
     */
    private final int kernelNum;

    /**
     * カーネルサイズ
     */
    private final int kernelSize;

    /**
     * パディングサイズ
     */
    private final int paddingSize;

    /**
     * ストライド
     */
    private final int stride;

    /**
     * 畳み込みパラメータ
     */
    private final ConvolutionParameter convolutionParameter;

    public SimpleConvolutionNet(int inputWidth, int inputHeight, int hiddenSize, int outputSize, ActivationType activationType, WeightInitializeType weightInitializeType, int kernelNum, int kernelSize, int paddingSize, int stride) {
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.activationType = activationType;
        this.weightInitializeType = weightInitializeType;
        this.kernelNum = kernelNum;
        this.kernelSize = kernelSize;
        this.paddingSize = paddingSize;
        this.stride = stride;
        this.convolutionParameter = new ConvolutionParameter(inputWidth, inputHeight, kernelSize, kernelSize, paddingSize, paddingSize, stride, stride);


        this.initWeight();
        this.initLayers();
    }

    LinkedHashMap<String, Layer> layers;
    public HashMap<String, INDArray> params;
    LastLayer lastLayer;

    private void initWeight() {
        double weightInitStd = 0.01;

        this.params = new HashMap<>();
        this.params.put("W1", Nd4j.randn(new int[]{this.kernelNum, 1, this.kernelSize, this.kernelSize}).mul(weightInitStd));
        this.params.put("b1", Nd4j.zeros(this.kernelNum));

        int poolOutputSize = this.kernelNum * (this.convolutionParameter.getOutputWidth() / 2) * (this.convolutionParameter.getOutputHeight() / 2);
        this.params.put("W2", Nd4j.randn(poolOutputSize, this.hiddenSize).mul(weightInitStd));
        this.params.put("b2", Nd4j.zeros(this.hiddenSize));

        this.params.put("W3", Nd4j.randn(this.hiddenSize, this.outputSize).mul(weightInitStd));
        this.params.put("b3", Nd4j.zeros(this.outputSize));
    }

    private void initLayers() {
        this.layers = new LinkedHashMap<>();
        this.layers.put("Conv1", new ConvolutionLayer(this.convolutionParameter, this.params.get("W1"), this.params.get("b1")));
        this.layers.put("Relu1", new ReLULayer());
        this.layers.put("Pool1", new PoolingLayer(2, 2, 2, 0));
        this.layers.put("Affine1", new AffineLayer2(this.params.get("W2"), this.params.get("b2")));
        this.layers.put("Relu2", new ReLULayer());
        this.layers.put("Affine2", new AffineLayer2(this.params.get("W3"), this.params.get("b3")));
        this.lastLayer = new SoftmaxWithLossLayer();
    }

    public enum WeightInitializeType {
        ReLU,
        Sigmoid,
        He,
        Xavier
    }

    public enum ActivationType {
        ReLU,
        Sigmoid,
    }


    @Override
    public INDArray predict(INDArray x) {
        for (String key : this.layers.keySet()) {
            System.out.println("key : " + key);
            System.out.println(x.shapeInfoToString());
            x = layers.get(key).forward(x);
        }
        return x;
    }

    @Override
    public double loss(INDArray input, INDArray t) {
        INDArray y = this.predict(input);

        System.out.println("softmax input y : ");
        System.out.println(y);
        System.out.println(y.shapeInfoToString());

        System.out.println("softmax input t : ");
        System.out.println(t);
        System.out.println(t.shapeInfoToString());

        return this.lastLayer.forward(y, t);
    }

    @Override
    public double accuracy(INDArray x, INDArray t) {
        INDArray predicted = this.predict(x);
        predicted = Nd4j.argMax(predicted, 1);
        INDArray y = Nd4j.argMax(t, 1);

        double correctNumber = predicted.eq(y).sumNumber().doubleValue();
        long size = t.size(0);
        return correctNumber / size;
    }

    @Override
    public HashMap<String, INDArray> numericalGradient(INDArray x, INDArray t) {

        Function<INDArray, Double> lossFunc = w -> this.loss(x, t);
        HashMap<String, INDArray> grads = new HashMap<>();
        for (int i = 1; i <= 3; i++) {
            grads.put("W" + i, NdUtil.NumericalGradient(lossFunc, this.params.get("W" + i)));
            grads.put("b" + i, NdUtil.NumericalGradient(lossFunc, this.params.get("b" + i)));
        }
        return grads;
    }

    @Override
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
        ConvolutionLayer convolutionLayer1 = (ConvolutionLayer) this.layers.get("Conv1");
        grads.put("W1", convolutionLayer1.dWeight);
        grads.put("b1", convolutionLayer1.dBias);

        AffineLayer2 affineLayer1 = (AffineLayer2) this.layers.get("Affine1");
        grads.put("W2", affineLayer1.dWeight);
        grads.put("b2", affineLayer1.dBias);

        AffineLayer2 affineLayer2 = (AffineLayer2) this.layers.get("Affine2");
        grads.put("W3", affineLayer2.dWeight);
        grads.put("b3", affineLayer2.dBias);

        return grads;
    }

    @Override
    public HashMap<String, INDArray> getParams() {
        return this.params;
    }
}
