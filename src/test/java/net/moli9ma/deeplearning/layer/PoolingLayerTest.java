package net.moli9ma.deeplearning.layer;

import net.moli9ma.deeplearning.ConvolutionUtil;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class PoolingLayerTest {

    @Test
    void test1() {

        int poolWidth = 2;
        int poolHeight = 2;
        PoolingLayer poolingLayer = new PoolingLayer(poolWidth, poolHeight);

        // forward
        INDArray dataA = Nd4j.create(new double[][]{
                {1, 1, 1},
                {1, 1, 1},
                {1, 1, 1},
        });

        INDArray dataB = Nd4j.create(new double[][]{
                {2, 2, 2},
                {2, 2, 2},
                {2, 2, 2},
        });

        INDArray dataC = Nd4j.create(new double[][]{
                {3, 3, 3},
                {3, 3, 3},
                {3, 3, 3},
        });

        INDArray dataD = Nd4j.create(new double[][]{
                {4, 4, 4},
                {4, 4, 4},
                {4, 4, 4},
        });

        INDArray input = Nd4j.create(new int[]{2, 2, 3, 3}, 'c');
        input.put(new INDArrayIndex[]{point(0), point(0), all(), all()}, dataA);
        input.put(new INDArrayIndex[]{point(0), point(1), all(), all()}, dataB);
        input.put(new INDArrayIndex[]{point(1), point(0), all(), all()}, dataC);
        input.put(new INDArrayIndex[]{point(1), point(1), all(), all()}, dataD);
        poolingLayer.forward(input);

        // backward
    }
}
