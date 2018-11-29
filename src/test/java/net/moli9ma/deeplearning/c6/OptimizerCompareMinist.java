package net.moli9ma.deeplearning.c6;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;
import net.moli9ma.deeplearning.MnistLoader;
import net.moli9ma.deeplearning.MultiLayerNet;
import net.moli9ma.deeplearning.optimizer.*;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class OptimizerCompareMinist extends Application {

    @Test
    void hoge () {
        int x = (int) 10l;

        HashMap<String, Optimizer> optimizers = new HashMap<>();
        optimizers.put("SGD", new StochasticGradientDescentOptimizer());
        optimizers.put("Momentum", new MomentumOptimizer());
        optimizers.put("AdaGrad", new AdaGradOptimizer());
        //optimizers.put("Adam", new AdamOptimizer());

        try {
            HashMap<String, List<Double>> losses = new OptimizerCompareMinist().getData(optimizers);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }


    public HashMap<String, List<Double>> getData(HashMap<String, Optimizer> optimizers) throws IOException {

        // Mnist画像を読み込む
        MnistLoader train = new MnistLoader(MnistLoader.TrainImages, MnistLoader.TrainLabels);
        INDArray x_train = train.normalizedImages();
        INDArray t_train = train.oneHotLabels();

        // パラメータ初期化
        int batchSize = 128;
        int maxIterations = 2000;
        HashMap<String, List<Double>> losses = new HashMap<>();
        for (String key : optimizers.keySet()){
            losses.put(key, new ArrayList<>());
        }

        // 比較対象のOptimizerに対してそれぞれnetworkを初期化する
        HashMap<String, MultiLayerNet> networks = new HashMap<>();
        List<Long> hiddenSizeList = new ArrayList<>();
        hiddenSizeList.add(100L);
        hiddenSizeList.add(100L);
        hiddenSizeList.add(100L);
        hiddenSizeList.add(100L);
        for (String key : optimizers.keySet()) {
            networks.put(key, new MultiLayerNet(784, 10, hiddenSizeList, MultiLayerNet.ActivationType.ReLU));
        }

        DataSet dataset = new DataSet(x_train, t_train);

        // 学習
        for (int i = 0; i < maxIterations; i++) {

            DataSet sample = dataset.sample(batchSize);
            INDArray x_batch = sample.getFeatures();
            INDArray t_batch = sample.getLabels();

            for (String key : optimizers.keySet()) {
                HashMap<String, INDArray> grads = networks.get(key).gradient(x_batch, t_batch);
                optimizers.get(key).update(networks.get(key).params, grads);

                double loss = networks.get(key).loss(x_batch, t_batch);
                losses.get(key).add(loss);
            }

            if (i % 100 == 0) {
                System.out.println("iteration : " + i);
                for (String key : optimizers.keySet()) {
                    double loss = networks.get(key).loss(x_batch, t_batch);
                    System.out.println(key + " : " + loss);
                }
            }
        }
        return losses;
    }


    @Override
    public void start(Stage primaryStage) throws Exception {

        // 比較対象のOptimizerを初期化する
        HashMap<String, Optimizer> optimizers = new HashMap<>();
        optimizers.put("SGD", new StochasticGradientDescentOptimizer());
        optimizers.put("Momentum", new MomentumOptimizer());
        optimizers.put("AdaGrad", new AdaGradOptimizer());
        //optimizers.put("Adam", new AdamOptimizer());

        HashMap<String, List<Double>> losses = getData(optimizers);

        // plot
        primaryStage.setTitle("optimizer compare native");
        final NumberAxis xAxis = new NumberAxis();
        final NumberAxis yAxis = new NumberAxis();
        final LineChart<Number, Number> lineChart = new LineChart<Number, Number>(xAxis, yAxis);

        for (Map.Entry<String, Optimizer> entry : optimizers.entrySet()) {
            XYChart.Series series = new XYChart.Series();
            series.setName(entry.getKey());

            List<Double> xHistory = losses.get(entry.getKey());
            for (int i = 0; i < xHistory.size(); i++) {
                series.getData().add(new XYChart.Data<>((double)i + 1, xHistory.get(i)));
            }
            lineChart.getData().add(series);
        }

        Scene scene = new Scene(lineChart, 800, 600);
        primaryStage.setScene(scene);
        primaryStage.show();
    }
}