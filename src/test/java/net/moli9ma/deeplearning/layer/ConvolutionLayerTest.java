package net.moli9ma.deeplearning.layer;

import net.moli9ma.deeplearning.ConvolutionParameter;
import net.moli9ma.deeplearning.ConvolutionUtil;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class ConvolutionLayerTest {

    @Test
    void test1() {


        // miniBatch = 1, depth = 1
        {
            System.out.println("miniBatch = 1, depth = 1");
            int miniBatch = 1;
            int depth = 1;
            int height = 3;
            int width = 3;

            int kH = 2;
            int kW = 2;
            int sX = 1;
            int sY = 1;
            int pX = 0;
            int pY = 0;

            ConvolutionParameter parameter = new ConvolutionParameter(height, width, kH, kW, pY, pX, sX, sY);

            // kernel
            // kernel patternA
            INDArray kernelA = Nd4j.create(new double[][]{
                    {1, 1},
                    {1, 1},
            });

            INDArray kernel = Nd4j.create(new int[]{miniBatch, depth, kH, kW}, 'c');
            kernel.put(new INDArrayIndex[]{point(0), point(0), all(), all()}, kernelA);

            INDArray data = Nd4j.create(new double[][]{
                    {1, 2, 3},
                    {4, 5, 6},
                    {7, 8, 9},
            });

            INDArray input = Nd4j.create(new int[]{miniBatch, depth, height, width}, 'c');
            input.put(new INDArrayIndex[]{point(0), point(0), all(), all()}, data);


            INDArray bias = Nd4j.zeros(1);

            ConvolutionLayer convolutionLayer = new ConvolutionLayer(parameter, kernel, bias);

            INDArray out = convolutionLayer.forward(input);
            INDArray dx = convolutionLayer.backward(out);
            System.out.println(dx);
        }
    }

    @Test
    void test2() {

        {
            System.out.println("miniBatch = 2, depth = 2");
            int miniBatch = 2;
            int depth = 2;
            int height = 3;
            int width = 3;

            int kH = 2;
            int kW = 2;
            int sX = 1;
            int sY = 1;
            int pX = 0;
            int pY = 0;

            ConvolutionParameter parameter = new ConvolutionParameter(height, width, kH, kW, pY, pX, sX, sY);

            // kernel
            // kernel patternA
            INDArray kernelA = Nd4j.create(new double[][]{
                    {1, 2},
                    {3, 4},
            });

            INDArray kernelB = Nd4j.create(new double[][]{
                    {1, 2},
                    {3, 4},
            });

            INDArray kernel = Nd4j.create(new int[]{miniBatch, depth, kH, kW}, 'c');
            kernel.put(new INDArrayIndex[]{point(0), point(0), all(), all()}, kernelA);
            kernel.put(new INDArrayIndex[]{point(0), point(1), all(), all()}, kernelB);
            kernel.put(new INDArrayIndex[]{point(1), point(0), all(), all()}, kernelA);
            kernel.put(new INDArrayIndex[]{point(1), point(1), all(), all()}, kernelB);

            INDArray data = Nd4j.create(new double[][]{
                    {1, 2, 3},
                    {4, 5, 6},
                    {7, 8, 9},
            });

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

            INDArray input = Nd4j.create(new int[]{miniBatch, depth, height, width}, 'c');
            input.put(new INDArrayIndex[]{point(0), point(0), all(), all()}, dataA);
            input.put(new INDArrayIndex[]{point(0), point(1), all(), all()}, dataB);
            input.put(new INDArrayIndex[]{point(1), point(0), all(), all()}, dataC);
            input.put(new INDArrayIndex[]{point(1), point(1), all(), all()}, dataD);

            INDArray bias = Nd4j.zeros(1);
            ConvolutionLayer convolutionLayer = new ConvolutionLayer(parameter, kernel, bias);

            INDArray out = convolutionLayer.forward(input);
            INDArray dx = convolutionLayer.backward(out);
            System.out.println(dx);
        }
    }
}
