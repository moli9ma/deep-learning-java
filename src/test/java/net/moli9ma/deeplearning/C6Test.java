package net.moli9ma.deeplearning;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;
import net.moli9ma.deeplearning.optimizer.StochasticGradientDescentOptimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.function.Function;

public class C6Test extends Application {


    public static void main(String[] args) {
        launch(args);
    }

    public List<INDArray> getDatas() {
        Function<INDArray, Double> getGradient = x -> {
            double x1 = x.getDouble(0);
            double x2 = x.getDouble(1);
            double result = Math.pow(x1, 2)/20  + Math.pow(x2, 2);
            return result;
        };

        int iterateNum = 1000;

        INDArray parameter = Nd4j.create(new double[]{10.0, 4.0});
        //INDArray parameter = Nd4j.rand(2, 1);
        HashMap<String, INDArray> params = new HashMap<>();
        params.put("w1", parameter);

        StochasticGradientDescentOptimizer descent = new StochasticGradientDescentOptimizer(0.2);


        List<INDArray> results = new ArrayList<>();

        for (int i = 0; i < iterateNum; i++) {
            INDArray grad = NdUtil.NumericalGradient(getGradient, params.get("w1"));
            HashMap<String, INDArray> grads = new HashMap<>();
            grads.put("w1", grad);
            descent.update(params, grads);



            INDArray output = Nd4j.create(new double[]{0,0});
            Nd4j.copy(params.get("w1"), output);
            results.add(output);
        }
        return results;
    }



    @Override
    public void start(Stage stage) throws Exception {
        stage.setTitle("Line Chart Sample");
        //defining the axes
        final NumberAxis xAxis = new NumberAxis();
        final NumberAxis yAxis = new NumberAxis();
        xAxis.setLabel("Number of Month");

        //creating the chart
        final LineChart<Number,Number> lineChart = new LineChart<Number,Number>(xAxis,yAxis);

        lineChart.setTitle("Stock Monitoring, 2010");
        //defining a series
        XYChart.Series series = new XYChart.Series();
        series.setName("My portfolio");
        //populating the series with data

        for (INDArray result : getDatas()) {
            series.getData().add(new XYChart.Data(result.getDouble(0), result.getDouble(1)));
        }

        Scene scene  = new Scene(lineChart,800,600);
        lineChart.getData().add(series);

        stage.setScene(scene);
        stage.show();

    }
}
