package net.moli9ma.deeplearning;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class SigmoidLayerTest {

    @Test
    void t1() {
        INDArray x = Nd4j.create(new double[]{0.1, 1.0, -5.0, 3.0});
        SigmoidLayer sigmoidLayer = new SigmoidLayer();
        INDArray f = sigmoidLayer.forward(x);
        assertEquals("[[    0.5250,    0.7311,    0.0067,    0.9526]]", f.toString());

        INDArray dout = Nd4j.create(new double[]{0.1, 0.5, 0.1, 0.3});
        INDArray bf = sigmoidLayer.backward(dout);
        assertEquals("[[    0.0249,    0.0983,    0.0007,    0.0136]]", bf.toString());
    }
}


