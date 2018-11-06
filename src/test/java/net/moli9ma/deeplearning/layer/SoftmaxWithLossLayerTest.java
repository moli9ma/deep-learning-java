package net.moli9ma.deeplearning.layer;

import net.moli9ma.deeplearning.layer.SoftmaxWithLossLayer;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class SoftmaxWithLossLayerTest {

    @Test
    void t1() {
        {

            INDArray x = Nd4j.create(new double[]{0.3, 0.2, 0.5});
            INDArray t = Nd4j.create(new double[]{0, 1, 0});
            SoftmaxWithLossLayer layer = new SoftmaxWithLossLayer();
            double f = layer.forward(x, t);
            assertEquals("[[    0.3199,    0.2894,    0.3907]]", layer.y.toString());
            assertEquals(1.2398308515548706, f);

            INDArray b = layer.backward();
            assertEquals("[[    0.3199,   -0.7106,    0.3907]]", b.toString());

        }

        {

            INDArray x = Nd4j.create(new double[]{0.01, 4.2, 0.1});
            INDArray t = Nd4j.create(new double[]{0, 1, 0});
            SoftmaxWithLossLayer layer = new SoftmaxWithLossLayer();
            double f = layer.forward(x, t);

/*
            System.out.println(layer.y);
            System.out.println(f);
*/

            assertEquals("[[    0.0147,    0.9693,    0.0161]]", layer.y.toString());
            assertEquals(0.031226160004734993, f);

            INDArray b = layer.backward();
            assertEquals("[[    0.0147,   -0.0307,    0.0161]]", b.toString());
            //System.out.print(b);
        }
    }
}
