package net.moli9ma.deeplearning;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TwoLayerNetTest {

    @Test
    public void testAccuracy() {

        {

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
    }
}
