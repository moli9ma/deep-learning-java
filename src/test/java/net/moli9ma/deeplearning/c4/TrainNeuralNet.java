package net.moli9ma.deeplearning.c4;

import net.moli9ma.deeplearning.MnistLoader;
import net.moli9ma.deeplearning.TwoLayerNet;
import net.moli9ma.deeplearning.TwolayerNetParameter;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TrainNeuralNet {

    @Test
    public void miniBatch() throws IOException {


        MnistLoader train = new MnistLoader(MnistLoader.TrainImages, MnistLoader.TrainLabels);
        INDArray x_train = train.normalizedImages();
        INDArray t_train = train.oneHotLabels();

        /*assertArrayEquals(new int[] {60000, 784}, x_train.shape());
        assertArrayEquals(new int[] {60000, 10}, t_train.shape());*/

        List<Double> train_loss_list =  new ArrayList<>();
        int iters_num = 10000;

        // int train_size = images.size(0);
        int batch_size = 100;
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
