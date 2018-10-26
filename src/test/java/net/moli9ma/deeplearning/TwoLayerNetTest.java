package net.moli9ma.deeplearning;

import jdk.nashorn.internal.ir.annotations.Ignore;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TwoLayerNetTest {

    @Test
    public void testAccuracy() {
        TwoLayerNet net = new TwoLayerNet(
                Nd4j.create(new double[][] {{1, 1}, {1, 1}}),
                Nd4j.create(new double[] {1, 1}),
                Nd4j.create(new double[][] {{1, 1}, {1, 1}}),
                Nd4j.create(new double[] {1, 1}));
        INDArray x = Nd4j.create(new double[][] {{1, 0}, {0, 1}});
        INDArray t = Nd4j.create(new double[][] {{1, 0}, {0, 1}});
        double accuracy = net.accuracy(x, t);
        assertEquals(0.5, accuracy, 5e-6);
    }

    @Test
    public void numericalGradient() {

        TwoLayerNet net = new TwoLayerNet(
                Nd4j.create(new double[][] {{1, 1}, {1, 1}}),
                Nd4j.create(new double[] {1, 1}),
                Nd4j.create(new double[][] {{1, 1}, {1, 1}}),
                Nd4j.create(new double[] {1, 1}));
        INDArray x = Nd4j.create(new double[][] {{1, 0}, {0, 1}});
        INDArray t = Nd4j.create(new double[][] {{1, 0}, {0, 1}});

        for (int i = 0; i < 10; i++) {

            net.numericalGradient(x, t);
            System.out.println("gradient = " + i);

/*
            System.out.println(net.weight1);
            System.out.println(net.bias1);
            System.out.println(net.weight2);
            System.out.println(net.bias2);
*/
        }
    }

    @Test
    public void testConstructor() {
        TwoLayerNet net = new TwoLayerNet(784, 100, 10, 0.01);
        assertArrayEquals(new long[] {784, 100}, net.parameter.weight1.shape());
        assertArrayEquals(new long[] {1, 100}, net.parameter.bias1.shape());
        assertArrayEquals(new long[] {100, 10}, net.parameter.weight2.shape());
        assertArrayEquals(new long[] {1, 10}, net.parameter.bias2.shape());
    }


    @Ignore
    @Test
    public void miniBatch() throws IOException {


        MnistLoader train = new MnistLoader(MnistLoader.TrainImages, MnistLoader.TrainLabels);
        INDArray x_train = train.normalizedImages();
        INDArray t_train = train.oneHotLabels();

        /*assertArrayEquals(new int[] {60000, 784}, x_train.shape());
        assertArrayEquals(new int[] {60000, 10}, t_train.shape());*/

        List<Double> train_loss_list =  new ArrayList<>();
        int iters_num = 100;

        // int train_size = images.size(0);
        int batch_size = 10;
        double learning_rate = 0.1;
        TwoLayerNet network = new TwoLayerNet(784, 50, 10, 0.01);


        // batch_size分のデータをランダムに取り出します。
        for (int i = 0; i < iters_num; ++i) {
            long start = System.currentTimeMillis();

            // ミニバッチの取得
            DataSet ds = new DataSet(x_train, t_train);
            DataSet sample = ds.sample(batch_size);
            INDArray x_batch = sample.getFeatures();
            INDArray t_batch = sample.getLabels();

            TwolayerNetParameter grad =  network.numericalGradient(x_batch, t_batch);

            network.parameter.weight1 = network.parameter.weight1.sub(grad.weight1.mul(learning_rate));
            network.parameter.bias1 = network.parameter.bias1.sub(grad.bias1.mul(learning_rate));

            network.parameter.weight2 = network.parameter.weight2.sub(grad.weight2.mul(learning_rate));
            network.parameter.bias2 = network.parameter.bias2.sub(grad.bias2.mul(learning_rate));


            // 学習経過の記録
            double loss = network.loss(x_batch, t_batch);
            train_loss_list.add(loss);
            System.out.printf("iteration %d loss=%f elapse=%dms%n", i, loss, System.currentTimeMillis() - start);
        }
    }

}
