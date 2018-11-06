package net.moli9ma.deeplearning.c6;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.CategoryAxis;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;
import net.moli9ma.deeplearning.NdUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

public class weightInitActivationHistogram extends Application {

    public static void main(String[] args) {
        launch(args);
    }

    // label : count
    public List<Double> getCounts(INDArray x) {
        List<Double> casted = new ArrayList<>();
        for (int i = 0; i < x.rows(); i++) {
            for (int j = 0; j < x.columns(); j++) {
                casted.add(BigDecimal.valueOf(x.getDouble(i, j)).setScale(1, RoundingMode.DOWN).doubleValue());
            }
        }
        return casted;
    }

    @Override
    public void start(Stage primaryStage) throws Exception {

        // 1000個のデータ
        final INDArray inputData = Nd4j.randn(new int[]{1000, 100});

        INDArray x = inputData;

        // 隠れ層のノード数
        final int nodeNum = 100;

        // 隠れ層の数
        final int hiddenLayerSize = 3;

        // アクティベーションの結果
        List<INDArray> activations = new ArrayList<>();

        for (int i = 0; i < hiddenLayerSize; i++) {

            if (i != 0) {
                x = activations.get(i - 1);
            }

            //INDArray w = Nd4j.randn(new int[]{nodeNum, nodeNum}).mul(0.01);
            INDArray w = Nd4j.randn(new int[]{nodeNum, nodeNum}).mul(Math.sqrt(1.0 / nodeNum));

            INDArray a = x.mmul(w);

            INDArray z = NdUtil.Sigmoid(a);

            activations.add(z);
        }

        // 系列を生成
        List<XYChart.Series<String, Number>> seriesList = new ArrayList<>();

        for (int i = 0; i < activations.size(); i++) {

            final String seriesName = String.valueOf(i);
            XYChart.Series<String, Number> series = new XYChart.Series<>();
            series.setName(seriesName);

            getCounts(activations.get(i)).stream().collect(Collectors.groupingBy(
                    Double:: doubleValue,
                    Collectors.counting()
            )).entrySet().stream().sorted(java.util.Map.Entry.comparingByKey()).forEach(
                    e -> {
                        series.getData().add(new XYChart.Data<>(e.getKey().toString(), e.getValue()));
                    }
            );
            seriesList.add(series);
        }

        // 軸を作成
        NumberAxis numberAxis = new NumberAxis();
        numberAxis.setLabel("frequency");
        CategoryAxis categoryAxis = new CategoryAxis();
        categoryAxis.setLabel("score");

        // 棒グラフを生成
        BarChart<String, Number> bc = new BarChart<>(categoryAxis, numberAxis);
        //bc.setBarGap(10);
        //bc.setCategoryGap(10);
        bc.getData().addAll(seriesList);

        // Scene
        Scene scene = new Scene(bc, 400, 400);

        // Stage
        primaryStage.setTitle("得点分布図");
        primaryStage.setScene(scene);
        primaryStage.show();
        primaryStage.show();
    }


}
