package net.moli9ma.deeplearning;

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
}
