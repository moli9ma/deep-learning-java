package net.moli9ma.deeplearning.c7;

import net.moli9ma.deeplearning.SimpleConvolutionNet;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;

public class GradientCheck {

    @Test
    void test() {

        int inputWidth = 10;
        int inputHeight = 10;
        int hiddenSize = 10;
        int outputSize = 10;

        // conv parameter
        int kernelNum = 10;
        int kernelSize = 3;
        int padding = 0;
        int stride = 1;

        SimpleConvolutionNet network = new SimpleConvolutionNet(
                inputWidth,
                inputHeight,
                hiddenSize,
                outputSize,
                SimpleConvolutionNet.ActivationType.ReLU,
                SimpleConvolutionNet.WeightInitializeType.ReLU,
                kernelNum,
                kernelSize,
                padding,
                stride
                );

        // 入力データ
        INDArray x = Nd4j.randn(new int[]{1, 1, 10, 10});
        System.out.println(x.shapeInfoToString());

        // 教師データ
        INDArray t = Nd4j.create(new double[]{1}).reshape(1, 1);
        System.out.println(t.shapeInfoToString());

        //HashMap<String, INDArray> grandNum = network.numericalGradient(x, t);
        HashMap<String, INDArray> grand = network.gradient(x, t);

        for (String key : grand.keySet()) {

            System.out.println("key : " + key);
          //  System.out.println("grandNum : " + grandNum.get(key));
            System.out.println("grand : " + grand.get(key));
        }
    }
}
