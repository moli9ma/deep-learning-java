package net.moli9ma.deeplearning.c5;

import net.moli9ma.deeplearning.MnistLoader;
import net.moli9ma.deeplearning.NdUtil;
import net.moli9ma.deeplearning.TwoLayerNetBackPropagation;
import net.moli9ma.deeplearning.TwolayerNetParameter;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class GradientCheck {

    @Test
    void gradientCheck() throws Exception {
        MnistLoader train = new MnistLoader(MnistLoader.TrainImages, MnistLoader.TrainLabels);
        INDArray x_train = train.normalizedImages();
        INDArray t_train = train.oneHotLabels();

        TwoLayerNetBackPropagation network = new TwoLayerNetBackPropagation(784, 50, 10, 0.01);

        // ミニバッチの取得
        int batch_size = 1;
        DataSet ds = new DataSet(x_train, t_train);
        DataSet sample = ds.sample(batch_size);
        INDArray x_batch = sample.getFeatures();
        INDArray t_batch = sample.getLabels();

        TwolayerNetParameter parameterByNumericalGradient = network.numericalGradient(x_batch, t_batch);
        TwolayerNetParameter parameterByGradient = network.gradient(x_batch, t_batch);

        double diffWeight1 = NdUtil.average(Transforms.abs(parameterByGradient.weight1.sub(parameterByNumericalGradient.weight1)));
        double diffBias1 = NdUtil.average(Transforms.abs(parameterByGradient.bias1.sub(parameterByNumericalGradient.bias1)));
        double diffWeight2 = NdUtil.average(Transforms.abs(parameterByGradient.weight2.sub(parameterByNumericalGradient.weight2)));
        double diffBias2 = NdUtil.average(Transforms.abs(parameterByGradient.bias2.sub(parameterByNumericalGradient.bias2)));

        /*
        b1:9.70418809871e-13
        W2:8.41139039497e-13
        b2:1.1945999745e-10
        W1:2.2232446644e-13
*/


        // pythonよりだいぶ誤差おおきい...
/*
        System.out.println(diffWeight1);
        System.out.println(diffBias1);
        System.out.println(diffWeight2);
        System.out.println(diffBias2);
        5.508259242894698E-5
        5.725093558430672E-4
        4.843713343143463E-4
        5.594544112682342E-4
*/

        assertTrue(diffWeight1 < 1e-3);
        assertTrue(diffBias1 < 1e-3);
        assertTrue(diffWeight2 < 1e-3);
        assertTrue(diffBias2 < 1e-3);
    }
}
