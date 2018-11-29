package net.moli9ma.deeplearning;

import net.moli9ma.deeplearning.layer.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * 単純な畳み込みネットワーク
 * conv - relu - pool - affine - relu - affine - softmax
 *
 *
 *
 *
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
     * 隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
     */
    private final List<Integer> hiddenSizeList;

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
     *             'relu'または'he'を指定した場合は「Heの初期値」を設定
     *             'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
     */
    private final WeightInitializeType weightInitializeType;

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

    public SimpleConvolutionNet(int inputWidth, int inputHeight, List<Integer> hiddenSizeList, int outputSize, ActivationType activationType, WeightInitializeType weightInitializeType, int kernelSize, int paddingSize, int stride) {
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.hiddenSizeList = hiddenSizeList;
        this.outputSize = outputSize;
        this.activationType = activationType;
        this.weightInitializeType = weightInitializeType;
        this.kernelSize = kernelSize;
        this.paddingSize = paddingSize;
        this.stride = stride;
        this.convolutionParameter = new ConvolutionParameter(inputWidth, inputHeight, kernelSize, kernelSize, paddingSize, paddingSize, stride, stride);
    }

/*

            # 重みの初期化
    self.params = {}
    self.params['W1'] = weight_init_std * \
            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
    self.params['b1'] = np.zeros(filter_num)
    self.params['W2'] = weight_init_std * \
            np.random.randn(pool_output_size, hidden_size)
    self.params['b2'] = np.zeros(hidden_size)
    self.params['W3'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
    self.params['b3'] = np.zeros(output_size)

            # レイヤの生成
    self.layers = OrderedDict()
    self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                       conv_param['stride'], conv_param['pad'])
    self.layers['Relu1'] = Relu()
    self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
    self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
    self.layers['Relu2'] = Relu()
    self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

    self.last_layer = SoftmaxWithLoss()
*/

    LinkedHashMap<String, Layer> layers;
    public HashMap<String, INDArray> params;
    LastLayer lastLayer;

    private void initWeight() {

        int filterNum = 10;
        double weightInitStd = 0.01;

        this.params = new HashMap<>();
        // for convolution layer
        this.params.put("W1", Nd4j.randn(new int[]{filterNum, 1, this.inputWidth, this.inputHeight}).mul(weightInitStd));
        this.params.put("b1", Nd4j.zeros(filterNum));

        // for pooling layer
/*
        this.params.put("W2", Nd4j.randn().mul(weightInitStd));
        this.params.put("b2", Nd4j.zeros(filterNum));
*/

        this.params.put("W3", Nd4j.randn(this.hiddenSizeList.size() , this.convolutionParameter.getOutputNum()).mul(weightInitStd));
        this.params.put("b3", Nd4j.zeros(filterNum));

    }


    private void initLayers() {
        this.layers = new LinkedHashMap<>();

/*
        this.layers.put("Conv1", );
        this.layers.put("Relu1", new ReLULayer());
        this.layers.put("Pool1", new );
        this.layers.put("Affine1", new AffineLayer());
        this.layers.put("Relu2", new ReLULayer());
        this.layers.put("Affine2", new AffineLayer());
*/
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
        return null;
    }

    @Override
    public double loss(INDArray input, INDArray t) {
        return 0;
    }

    @Override
    public double accuracy(INDArray x, INDArray t) {
        return 0;
    }

    @Override
    public HashMap<String, INDArray> numericalGradient(INDArray x, INDArray t) {
        return null;
    }

    @Override
    public HashMap<String, INDArray> gradient(INDArray x, INDArray t) {
        return null;
    }
}
