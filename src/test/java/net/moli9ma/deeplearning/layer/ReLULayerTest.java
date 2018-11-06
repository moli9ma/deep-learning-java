package net.moli9ma.deeplearning.layer;

import net.moli9ma.deeplearning.layer.ReLULayer;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ReLULayerTest {

    @Test
    void t1() {

        // forward
        INDArray x = Nd4j.create(new double[][]{
                {1.0, -0.5},
                {-2.0, 3.0}
        });
        INDArray fExpected = Nd4j.create(new double[][]{
                {1.0, 0},
                {0, 3.0}
        });

        ReLULayer reLULayer = new ReLULayer();
        INDArray f = reLULayer.forward(x);
        assertEquals(fExpected, f);


        // backward
        INDArray y = Nd4j.create(new double[][]{
                {-1.0, -0.5},
                {2.0, -3.0}
        });

        INDArray bExpected = Nd4j.create(new double[][]{
                {-1.0, 0},
                {0, -3.0}
        });

        INDArray b = reLULayer.backward(y);
        assertEquals(bExpected, b);

    }
}
