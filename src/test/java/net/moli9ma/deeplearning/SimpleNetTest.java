package net.moli9ma.deeplearning;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SimpleNetTest {


    @Test
    void SimpleNet() {

        INDArray weight = Nd4j.rand(2, 3);
        SimpleNet simpleNet = new SimpleNet(weight);

        INDArray input = Nd4j.create(new double[]{0.6, 0.9,});
        INDArray predicted = simpleNet.predict(input);

        INDArray t = Nd4j.create(new double[]{0, 0, 1});
        double loss = simpleNet.loss(input, t);

        System.out.println(predicted);
        System.out.println(loss);
    }
}
