package net.moli9ma.deeplearning;

import net.moli9ma.deeplearning.layer.AddLayer;
import net.moli9ma.deeplearning.layer.AddLayerData;
import net.moli9ma.deeplearning.layer.MulLayer;
import net.moli9ma.deeplearning.layer.MulLayerData;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class C5Test {

    @Test
    void t1() {


        double apple = 100;
        double appleNum = 2;
        double orange = 150;
        double orangeNum = 3;
        double tax = 1.1;

// layer
        MulLayer mulAppleLayer = new MulLayer();
        MulLayer mulOrangeLayer = new MulLayer();
        AddLayer addAppleOrangeLayer = new AddLayer();
        MulLayer mulTaxLayer = new MulLayer();

        // forward
        double applePrice = mulAppleLayer.forward(new MulLayerData(apple, appleNum));
        double orangePrice = mulOrangeLayer.forward(new MulLayerData(orange, orangeNum));

        double allPrice = addAppleOrangeLayer.foraward(new AddLayerData(applePrice, orangePrice));
        double price = mulTaxLayer.forward(new MulLayerData(allPrice, tax));

        // backward
        double dprice = 1;
        MulLayerData dMulTaxLayerData = mulTaxLayer.backward(dprice);
        double dallPrice = dMulTaxLayerData.x;
        double dTax = dMulTaxLayerData.y;

        AddLayerData data = addAppleOrangeLayer.backward(dallPrice);
        double dApplePrice = data.x;
        double dOrangePrice = data.y;

        MulLayerData MulOrangeLayerData = mulOrangeLayer.backward(dOrangePrice);
        double dOrange = MulOrangeLayerData.x;
        double dOrangeNum = MulOrangeLayerData.y;

        MulLayerData MulAppleLayerData = mulAppleLayer.backward(dApplePrice);
        double dApple = MulAppleLayerData.x;
        double dAppleNum = MulAppleLayerData.y;

        assertEquals(715, (int) price);

        assertEquals(2.2, dApple);
        assertEquals(110, (int) dAppleNum);

        assertEquals(3.3000000000000003, dOrange);
        assertEquals(165, (int) dOrangeNum);

        assertEquals(650, (int) dTax);
    }


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

    @Test
    void trainNeuralNet() throws Exception {

        MnistLoader train = new MnistLoader(MnistLoader.TrainImages, MnistLoader.TrainLabels);
        INDArray x_train = train.normalizedImages();
        INDArray t_train = train.oneHotLabels();
        MnistLoader test = new MnistLoader(MnistLoader.TestImages, MnistLoader.TestLabels);
        INDArray x_test = test.normalizedImages();
        INDArray t_test = test.oneHotLabels();

        List<Double> train_loss_list = new ArrayList<>();
        List<Double> train_acc_list = new ArrayList<>();
        List<Double> test_acc_list = new ArrayList<>();

        int iters_num = 10000;
        long train_size = x_train.size(0);
        int batch_size = 100;
        double learning_rate = 0.1;
        double iter_per_epoch = Math.max((train_size / batch_size), 1);
        // ミニバッチの取得
        DataSet ds = new DataSet(x_train, t_train);

        TwoLayerNetBackPropagation network = new TwoLayerNetBackPropagation(784, 50, 10, 0.01D);

        // batch_size分のデータをランダムに取り出します。
        for (int i = 0; i < iters_num; ++i) {

            DataSet sample = ds.sample(batch_size);
            INDArray x_batch = sample.getFeatures();
            INDArray t_batch = sample.getLabels();

            TwolayerNetParameter grad = network.gradient(x_batch, t_batch);
            network.parameter.weight1.subi(grad.weight1.mul(learning_rate));
            network.parameter.bias1.subi(grad.bias1.mul(learning_rate));
            network.parameter.weight2.subi(grad.weight2.mul(learning_rate));
            network.parameter.bias2.subi(grad.bias2.mul(learning_rate));

            // 学習経過の記録
            double loss = network.loss(x_batch, t_batch);
            train_loss_list.add(loss);

            if (i % iter_per_epoch == 0) {

                double trainAcc = network.accuracy(x_train, t_train);
                double testAcc = network.accuracy(x_test, t_test);
                train_acc_list.add(trainAcc);
                test_acc_list.add(testAcc);

                System.out.printf("iteration %d loss=%f %n", i, loss);
                System.out.printf("trainAccuracy = %f, testAccuracy = %f %n", trainAcc, testAcc);
            }
        }
    }
}
