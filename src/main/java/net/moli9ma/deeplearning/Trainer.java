package net.moli9ma.deeplearning;

import net.moli9ma.deeplearning.optimizer.Optimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * ニューラルネットの訓練を行うクラス
 */
public class Trainer {

    private final Network network;
    private final boolean verbose;
    private final Optimizer optimizer;

    private final INDArray xTrain;
    private final INDArray tTrain;

    private final INDArray xTest;
    private final INDArray tTest;

    private final int epochNumber;
    private final int batchNumber;

    private final Integer evaluateSampleNumberPerEpoch;

    // dependency property
    private final int trainNumber;
    private final int iterationNumberPerEpoch;
    private final int maxIterationNumber;

    List<Double> trainLossList = new ArrayList<>();
    List<Double> trainAccuracyList = new ArrayList<>();
    List<Double> testAccuracyList = new ArrayList<>();

    int currentIter = 0;
    int currentEpoch = 0;


    public Trainer(Network network, boolean verbose, Optimizer optimizer, INDArray xTrain, INDArray tTrain, INDArray xTest, INDArray tTest, int epochNumber, int batchNumber, Integer evaluateSampleNumberPerEpoch) {
        this.network = network;
        this.verbose = verbose;
        this.optimizer = optimizer;
        this.xTrain = xTrain;
        this.tTrain = tTrain;
        this.xTest = xTest;
        this.tTest = tTest;
        this.epochNumber = epochNumber;
        this.batchNumber = batchNumber;
        this.evaluateSampleNumberPerEpoch = evaluateSampleNumberPerEpoch;

        this.trainNumber = (int) xTrain.shape()[0];
        this.iterationNumberPerEpoch = Math.max(trainNumber / this.batchNumber, 1);
        this.maxIterationNumber = epochNumber * this.iterationNumberPerEpoch;
    }

    void trainStep() {
        DataSet dataset = new DataSet(this.xTrain, this.tTrain);
        DataSet sample = dataset.sample(this.batchNumber);
        INDArray xBatch = sample.getFeatures();
        INDArray tBatch = sample.getLabels();

        // 勾配を更新
        HashMap<String, INDArray> grads = this.network.gradient(xBatch, tBatch);
        this.optimizer.update(this.network.getParams(), grads);

        // 損失を計測 (verbose = trueでログを出力)
        double loss = this.network.loss(xBatch, tBatch);
        this.trainLossList.add(loss);
        if (this.verbose) {
            System.out.println("train loss : " + String.valueOf(loss));
        }

        // エポック毎の処理 (verbose = true でログを出力)
        if (this.currentIter % this.iterationNumberPerEpoch == 0) {
            this.currentEpoch += 1;

            if (evaluateSampleNumberPerEpoch != null) {
                // Trainの認識精度を計測
                DataSet trainSample = new DataSet(this.xTrain, this.tTrain).sample(evaluateSampleNumberPerEpoch);
                INDArray xTrainSample = trainSample.getFeatures();
                INDArray tTrainSample = trainSample.getLabels();
                double trainAccuracy = this.network.accuracy(xTrainSample, tTrainSample);
                this.trainAccuracyList.add(trainAccuracy);

                // Testの認識精度を計測
                DataSet testSample = new DataSet(this.xTest, this.tTest).sample(evaluateSampleNumberPerEpoch);
                INDArray xTestSample = testSample.getFeatures();
                INDArray tTestSample = testSample.getLabels();
                double testAccuracy = this.network.accuracy(xTestSample, tTestSample);
                this.testAccuracyList.add(testAccuracy);

                // TrainとTestの認識精度を計測
                if (this.verbose) {
                    System.out.println("epoch  :" + this.currentEpoch);
                    System.out.println("train accuracy : " + this.trainAccuracyList.toString());
                    System.out.println("test accuracy : " + this.testAccuracyList.toString());
                }
            }
        }

        // イテレーションを更新
        this.currentIter += 1;
    }

    public void train() {
        for (int i = 0; i < this.maxIterationNumber; i++) {
            this.trainStep();
        }
        double finalTestAccuracy = this.network.accuracy(this.xTest, this.tTest);
        if (this.verbose) {
            System.out.println("=============== Final Test Accuracy ===============");
            System.out.println("test acc:" + finalTestAccuracy);
        }
    }
}
