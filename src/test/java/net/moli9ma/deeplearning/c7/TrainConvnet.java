package net.moli9ma.deeplearning.c7;

import net.moli9ma.deeplearning.MnistLoader;
import net.moli9ma.deeplearning.SimpleConvolutionNet;
import net.moli9ma.deeplearning.Trainer;
import net.moli9ma.deeplearning.optimizer.AdamOptimizer;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;

public class TrainConvnet {

    @Test
    void test() throws IOException {

        MnistLoader train = new MnistLoader(MnistLoader.TrainImages, MnistLoader.TrainLabels);
        INDArray xTrain = train.normalizedImages();
        int xTrainNumber = (int) xTrain.shape()[0];
        xTrain = xTrain.reshape(new int[]{xTrainNumber, 1, 28, 28});

        INDArray tTrain = train.oneHotLabels();
        /*int tTrainNumber = (int) tTrain.shape()[0];
        tTrain = tTrain.reshape(new int[]{tTrainNumber, 1, 1, 10});*/

        MnistLoader test = new MnistLoader(MnistLoader.TestImages, MnistLoader.TestLabels);
        INDArray xTest = test.normalizedImages();
        xTest = xTest.reshape(new int[]{(int) xTest.shape()[0], 1, 28, 28});

        INDArray tTest = test.oneHotLabels();
/*        tTest = tTest.reshape(new int[]{(int) tTest.shape()[0], 1, 28, 28});*/

        int maxEpoch = 20;

        int inputWidth = 28;
        int inputHeight = 28;
        int hiddenSize = 100;
        int outputSize = 10;

        // conv parameter
        int kernelNum = 30;
        int kernelSize = 5;
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

        Trainer trainer = new Trainer(
                network,
                true,
                new AdamOptimizer(0.001),
                xTrain,
                tTrain,
                xTest,
                tTest,
                maxEpoch,
                100,
                1000);
        trainer.train();
    }
}