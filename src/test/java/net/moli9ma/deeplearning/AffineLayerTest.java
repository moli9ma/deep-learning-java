package net.moli9ma.deeplearning;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class AffineLayerTest {

    @Test
    void t1() {

        INDArray weight = Nd4j.create(new double[][]{{1, 1, 1}, {2, 2, 2}});
        INDArray bias = Nd4j.create(new double[]{1, 2, 3});

        INDArray x = Nd4j.create(new double[]{0.3, 0.7});
        //INDArray x = Nd4j.create(new double[][]{{0.1, 0.3}, {0.2, 0.4}});
        AffineLayer affineLayer = new AffineLayer(weight, bias);
        INDArray f = affineLayer.forward(x);
        assertEquals("[[    2.7000,    3.7000,    4.7000]]", f.toString());

        INDArray dout = Nd4j.create(new double[][]{{3, 4, 5}, {6, 7, 8}});
        INDArray d = affineLayer.backward(dout);
        assertEquals("[[   12.0000,   24.0000], \n" +
                " [   21.0000,   42.0000]]", d.toString());

        assertEquals("[[    5.1000,    6.1000,    7.1000]]", affineLayer.dWeight.toString());
        assertEquals("[[    9.0000,   11.0000,   13.0000]]", affineLayer.dBias.toString());
    }
}
