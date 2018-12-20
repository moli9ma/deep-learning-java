package net.moli9ma.deeplearning.c7;

import net.moli9ma.deeplearning.MnistLoader;
import net.moli9ma.deeplearning.SimpleConvolutionNet;
import net.moli9ma.deeplearning.Trainer;
import net.moli9ma.deeplearning.optimizer.AdamOptimizer;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;

public class TrainConvnet {

    @Test
    void test() throws IOException {

        MnistLoader train = new MnistLoader(MnistLoader.TrainImages, MnistLoader.TrainLabels);
        INDArray xTrain = train.normalizedImages();
        System.out.println(xTrain.shapeInfoToString());
        int xTrainNumber = (int) xTrain.shape()[0];
        xTrain = xTrain.reshape(new int[]{xTrainNumber, 1, 28, 28});
        xTrain = xTrain.get(new INDArrayIndex[]{NDArrayIndex.interval(0, 5000)});
        System.out.println(xTrain.shapeInfoToString());

        INDArray tTrain = train.oneHotLabels();
        tTrain = tTrain.get(new INDArrayIndex[]{NDArrayIndex.interval(0, 5000)});

        MnistLoader test = new MnistLoader(MnistLoader.TestImages, MnistLoader.TestLabels);
        INDArray xTest = test.normalizedImages();
        System.out.println(xTest.shapeInfoToString());
        xTest = xTest.reshape(new int[]{(int) xTest.shape()[0], 1, 28, 28});
        xTest = xTest.get(new INDArrayIndex[]{NDArrayIndex.interval(0, 1000)});
        System.out.println(xTest.shapeInfoToString());

        INDArray tTest = test.oneHotLabels();
        System.out.println(tTest.shapeInfoToString());
        tTest = tTest.get(new INDArrayIndex[]{NDArrayIndex.interval(0, 1000)});
        System.out.println(tTest.shapeInfoToString());

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
                new AdamOptimizer(0.003),
                xTrain,
                tTrain,
                xTest,
                tTest,
                maxEpoch,
                100,
                100);
        trainer.train();
    }
}