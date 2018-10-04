package net.moli9ma.deeplearning;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import static net.moli9ma.deeplearning.ConvolutionUtil.getPadded;
import static org.junit.jupiter.api.Assertions.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;


public class ConvolutionUtilTest {


    @Test
    void Im2col() {

        int kH = 2;
        int kW = 2;
        int sX = 1;
        int sY = 1;
        int pX = 0;
        int pY = 0;

        // input
        int height = 3;
        int width = 3;
        INDArray v1 = Nd4j.create(new double[][]{
                {1, 2, 3},
                {2, 3, 4},
                {3, 4, 5},
        });

        ConvolutionParameter parameter = new ConvolutionParameter(height, width, kH, kW, pY, pX, sX, sY);
        INDArray paddedInput = getPadded(v1, parameter);
        INDArray arr = ConvolutionUtil.Im2col(paddedInput, parameter);
        INDArray expected = Nd4j.create(new double[]{1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5}, new int[]{4, 4});
        assertEquals(expected, arr);
    }


    @Test
    void convolution4D() {

        int miniBatch = 2;
        int depth = 2;
        int height = 5;
        int width = 5;

        int kH = 3;
        int kW = 3;
        int sX = 1;
        int sY = 1;
        int pX = 0;
        int pY = 0;

        ConvolutionParameter parameter = new ConvolutionParameter(height, width, kH, kW, pY, pX, sX, sY);

        // kernel
        // kernel patternA
        INDArray kernelA = Nd4j.create(new double[][]{
                {1, 1, 1},
                {1, 1, 1},
                {1, 1, 1}
        });

        // kernel patternB
        INDArray kernelB = Nd4j.create(new double[][]{
                {2, 2, 2},
                {2, 2, 2},
                {2, 2, 2}
        });

        INDArray kernel = Nd4j.create(new int[]{miniBatch, depth, kH, kW}, 'c');
        kernel.put(new INDArrayIndex[]{point(0), point(0), all(), all()}, kernelA);
        kernel.put(new INDArrayIndex[]{point(0), point(1), all(), all()}, kernelB);
        kernel.put(new INDArrayIndex[]{point(1), point(0), all(), all()}, kernelA);
        kernel.put(new INDArrayIndex[]{point(1), point(1), all(), all()}, kernelB);

        //Input data: shape [miniBatch,depth,height,width]
        INDArray data = Nd4j.create(new double[][]{
                        {1, 2, 3, 4, 5},
                        {6, 7, 8, 9, 10},
                        {11, 12, 13, 14, 15},
                        {16, 17, 18, 19, 20},
                        {21, 22, 23, 24, 25}
                });

        INDArray input = Nd4j.create(new int[]{miniBatch, depth, height, width}, 'c');
        input.put(new INDArrayIndex[]{point(0), point(0), all(), all()}, data);
        input.put(new INDArrayIndex[]{point(0), point(1), all(), all()}, data);
        input.put(new INDArrayIndex[]{point(1), point(0), all(), all()}, data);
        input.put(new INDArrayIndex[]{point(1), point(1), all(), all()}, data);

        INDArray result = ConvolutionUtil.Convolution4D(input, kernel, miniBatch, depth, parameter);
        System.out.println(result);
    }
}
