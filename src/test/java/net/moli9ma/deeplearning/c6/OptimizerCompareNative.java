package net.moli9ma.deeplearning.c6;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import net.moli9ma.deeplearning.optimizer.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

public class OptimizerCompareNative extends Application {

    class History {
        List<Double> xHistory;
        List<Double> yHistory;

        public History(List<Double> xHistory, List<Double> yHistory) {
            this.xHistory = xHistory;
            this.yHistory = yHistory;
        }
    }


    @Override
    public void start(Stage primaryStage) throws Exception {

        BiFunction<Double, Double, Double> f = (x, y) -> Math.pow(x, 2) / 20.0 + Math.pow(y, 2);
        BiFunction<Double, Double, INDArray> df = (x, y) -> Nd4j.create(new double[]{x / 10.0, 2.0 * y});

        HashMap<String, Optimizer> optimizers = new HashMap<>();
        optimizers.put("SGD", new StochasticGradientDescentOptimizer(0.95));
        optimizers.put("Momentum", new MomentumOptimizer(0.1, 0.9));
        optimizers.put("AdaGrad", new AdaGradOptimizer(1.5));
        optimizers.put("Adam", new AdamOptimizer(0.3));

        HashMap<String, History> historis = new HashMap<>();

        // calc
        for (Map.Entry<String, Optimizer> entry : optimizers.entrySet()) {

            Optimizer optimizer = entry.getValue();

            // init
            INDArray initPosition = Nd4j.create(new double[]{-7.0, 2.0});
            HashMap<String, INDArray> params = new HashMap<>();
            params.put(entry.getKey(), initPosition);

            List<Double> xHistory = new ArrayList<>();
            List<Double> yHistory = new ArrayList<>();

            for (int i = 0; i < 30; i++) {
                Double x = params.get(entry.getKey()).getDouble(0);
                Double y = params.get(entry.getKey()).getDouble(1);
                xHistory.add(x);
                yHistory.add(y);

                HashMap<String, INDArray> grads = new HashMap<>();
                grads.put(entry.getKey(), df.apply(x, y));
                optimizer.update(params, grads);
            }

            historis.put(entry.getKey(), new History(xHistory, yHistory));
        }


        // plot
        primaryStage.setTitle("optimizer compare native");
        final NumberAxis xAxis = new NumberAxis();
        final NumberAxis yAxis = new NumberAxis();
        final LineChart<Number, Number> lineChart = new LineChart<Number, Number>(xAxis, yAxis);


        for (Map.Entry<String, Optimizer> entry : optimizers.entrySet()) {
            XYChart.Series series = new XYChart.Series();
            series.setName(entry.getKey());

            List<Double> xHistory = historis.get(entry.getKey()).xHistory;
            List<Double> yHistory = historis.get(entry.getKey()).yHistory;
            for (int i = 0; i < historis.get(entry.getKey()).xHistory.size(); i++) {
                series.getData().add(new XYChart.Data<>(xHistory.get(i), yHistory.get(i)));
            }
            lineChart.getData().add(series);
        }

        Scene scene = new Scene(lineChart, 800, 600);
        primaryStage.setScene(scene);
        primaryStage.show();
    }
}




/*
    def f(x, y):
        return x**2 / 20.0 + y**2


        def df(x, y):
        return x / 10.0, 2.0*y
*/

/*
        init_pos = (-7.0, 2.0)
        params = {}
        params['x'], params['y'] = init_pos[0], init_pos[1]
        grads = {}
        grads['x'], grads['y'] = 0, 0


        optimizers = OrderedDict()
        optimizers["SGD"] = SGD(lr=0.95)
        optimizers["Momentum"] = Momentum(lr=0.1)
        optimizers["AdaGrad"] = AdaGrad(lr=1.5)
        optimizers["Adam"] = Adam(lr=0.3)

        idx = 1

        for key in optimizers:
        optimizer = optimizers[key]
        x_history = []
        y_history = []
        params['x'], params['y'] = init_pos[0], init_pos[1]

        for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])

        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params, grads)


        x = np.arange(-10, 10, 0.01)
        y = np.arange(-5, 5, 0.01)

        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)

        # for simple contour line
        mask = Z > 7
        Z[mask] = 0

        # plot
        plt.subplot(2, 2, idx)
        idx += 1
        plt.plot(x_history, y_history, 'o-', color="red")
        plt.contour(X, Y, Z)
        plt.ylim(-10, 10)
        plt.xlim(-10, 10)
        plt.plot(0, 0, '+')
        #colorbar()
        #spring()
        plt.title(key)
        plt.xlabel("x")
        plt.ylabel("y")

        plt.show()*/
