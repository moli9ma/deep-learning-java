package net.moli9ma.deeplearning.c5;

import net.moli9ma.deeplearning.MnistLoader;
import net.moli9ma.deeplearning.TwoLayerNetBackPropagation;
import net.moli9ma.deeplearning.TwolayerNetParameter;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;
import java.util.List;

public class TrainNeuralNet {

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
