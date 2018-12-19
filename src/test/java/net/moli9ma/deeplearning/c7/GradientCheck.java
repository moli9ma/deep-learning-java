package net.moli9ma.deeplearning.c7;

import net.moli9ma.deeplearning.SimpleConvolutionNet;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.HashMap;

import static org.nd4j.linalg.indexing.NDArrayIndex.point;

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
        INDArray x = Nd4j.randn(new int[]{5, 1, 10, 10});
        System.out.println(x.shapeInfoToString());

        // 教師データ (one-hot)
        INDArray t = Nd4j.zeros(5, 10);
        t.put(new INDArrayIndex[]{point(0), point(0)}, 1);
        t.put(new INDArrayIndex[]{point(1), point(0)}, 1);
        t.put(new INDArrayIndex[]{point(2), point(0)}, 1);
        t.put(new INDArrayIndex[]{point(3), point(0)}, 1);
        t.put(new INDArrayIndex[]{point(4), point(0)}, 1);
        System.out.println(t.shapeInfoToString());

        //HashMap<String, INDArray> grandNum = network.numericalGradient(x, t);
        HashMap<String, INDArray> grand = network.gradient(x, t);
        //HashMap<String, INDArray> grandNum = network.numericalGradient(x, t);

        System.out.println("b1");
        System.out.println(grand.get("b1"));

        System.out.println("b2");
        System.out.println(grand.get("b2"));

        System.out.println("b3");
        System.out.println(grand.get("b3"));


        for (String key : grand.keySet()) {
            System.out.println("key : " + key);
            //INDArray result = Transforms.abs(grandNum.get(key).sub(grand.get(key)));
            //System.out.println("grandNum : " + result.mean());

            //System.out.println("grand : " + grand.get(key));
            System.out.println("grand shape: " + grand.get(key).shapeInfoToString());
        }
    }
}
