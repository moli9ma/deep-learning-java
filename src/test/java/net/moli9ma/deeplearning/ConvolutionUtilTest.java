package net.moli9ma.deeplearning;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import static org.junit.jupiter.api.Assertions.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;


public class ConvolutionUtilTest {


    @Test
    void Im2col() {

        {
            int kH = 2;
            int kW = 2;
            int sX = 1;
            int sY = 1;
            int pX = 0;
            int pY = 0;

            // input
            int height = 3;
            int width = 3;
            INDArray input = Nd4j.create(new double[][]{
                    {1, 2, 3},
                    {4, 5, 6},
                    {7, 8, 9},
            });

            ConvolutionParameter parameter = new ConvolutionParameter(width, height, kH, kW, pY, pX, sX, sY);
            INDArray arr = ConvolutionUtil.Im2col(input, parameter);
            INDArray expected = Nd4j.create(new double[][]{
                    {1, 2, 4, 5},
                    {2, 3, 5, 6},
                    {4, 5, 7, 8},
                    {5, 6, 8, 9}
            });
            assertEquals(expected, arr);
        }


        {
            int kH = 2;
            int kW = 2;
            int sX = 1;
            int sY = 1;
            int pX = 0;
            int pY = 0;

            // input
            int height = 3;
            int width = 4;
            INDArray input = Nd4j.create(new double[][]{
                    {1, 2, 3, 4},
                    {5, 6, 7, 8},
                    {9, 10, 11, 12},
            });

            ConvolutionParameter parameter = new ConvolutionParameter(width, height, kH, kW, pY, pX, sX, sY);
            INDArray arr = ConvolutionUtil.Im2col(input, parameter);
            INDArray expected = Nd4j.create(new double[][]{
                    {1, 2, 5, 6},
                    {2, 3, 6, 7},
                    {3, 4, 7, 8},
                    {5, 6, 9, 10},
                    {6, 7, 10, 11},
                    {7, 8, 11, 12}
            });
            assertEquals(expected, arr);
        }

        // padding
        {
            int kH = 3;
            int kW = 3;
            int sX = 1;
            int sY = 1;
            int pX = 1;
            int pY = 1;

            // input
            int height = 2;
            int width = 2;
            INDArray input = Nd4j.create(new double[][]{
                    {1, 2},
                    {3, 4}
            });

            ConvolutionParameter parameter = new ConvolutionParameter(width, height, kH, kW, pY, pX, sX, sY);
            INDArray arr = ConvolutionUtil.Im2col(input, parameter);

            INDArray expected = Nd4j.create(new double[][]{
                    {0, 0, 0, 0, 1.0000, 2.0000, 0, 3.0000, 4.0000},
                    {0, 0, 0, 1.0000, 2.0000, 0, 3.0000, 4.0000, 0},
                    {0, 1.0000, 2.0000, 0, 3.0000, 4.0000, 0, 0, 0},
                    {1.0000, 2.0000, 0, 3.0000, 4.0000, 0, 0, 0, 0}
            });
            assertEquals(expected, arr);
        }
    }

    @Test
    void col2im() {

        {
            int kH = 2;
            int kW = 2;
            int sX = 1;
            int sY = 1;
            int pX = 0;
            int pY = 0;

            // input
            int height = 3;
            int width = 3;
            INDArray input = Nd4j.create(new double[][]{
                    {1, 2, 3},
                    {4, 5, 6},
                    {7, 8, 9},
            });

            ConvolutionParameter parameter = new ConvolutionParameter(width, height, kH, kW, pY, pX, sX, sY);
            INDArray arr = ConvolutionUtil.Im2col(input, parameter);


            System.out.println("im2col:");
            System.out.println(arr);

            INDArray result = ConvolutionUtil.Col2Im(arr, parameter);
            System.out.print(result);

            //assertEquals(result);

        }


        {
            int kH = 2;
            int kW = 2;
            int sX = 1;
            int sY = 1;
            int pX = 0;
            int pY = 0;

            // input
            int height = 3;
            int width = 3;

            double[][] x1 = new double[][]{
                    {1, 2, 3},
                    {4, 5, 6},
                    {7, 8, 9},
            };

            double[][] x2 = new double[][]{
                    {1, 1, 1},
                    {1, 1, 1},
                    {1, 1, 1},
            };

            INDArray x = Nd4j.create(new double[][][][]{
                    {
                            x1, x2
                    },
                    {
                            x1, x2
                    }
            });

            INDArray input = Nd4j.create(new double[][]{
                    {1, 2, 4, 5, 1, 1, 1, 1,},
                    {2, 3, 5, 6, 1, 1, 1, 1,},
                    {4, 5, 7, 8, 1, 1, 1, 1,},
                    {5, 6, 8, 9, 1, 1, 1, 1,},
                    {1, 2, 4, 5, 1, 1, 1, 1,},
                    {2, 3, 5, 6, 1, 1, 1, 1,},
                    {4, 5, 7, 8, 1, 1, 1, 1,},
                    {5, 6, 8, 9, 1, 1, 1, 1,},
            });

            ConvolutionParameter parameter = new ConvolutionParameter(width, height, kH, kW, pY, pX, sX, sY);
            INDArray result = ConvolutionUtil.Col2Im(2, 2, input, parameter);
            System.out.print(result);
            //assertEquals(result);
        }
    }


    @Test
    void convolution2D() {

        {
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
            INDArray kernel = Nd4j.create(new double[][]{
                    {1, 1},
                    {1, 1}
            });

            //Input data: shape [miniBatch,depth,height,width]
            INDArray input = Nd4j.create(new double[][]{
                    {1, 2, 3},
                    {4, 5, 6},
                    {7, 8, 9}
            });

            INDArray result = ConvolutionUtil.Convolution2D(input, kernel, parameter);

            INDArray expected = Nd4j.create(new double[][]{
                    {12, 16},
                    {24, 28}
            });

            assertEquals(expected, result);
        }

        // padding
        {
            int height = 2;
            int width = 2;

            int kH = 2;
            int kW = 2;
            int sX = 1;
            int sY = 1;
            int pX = 1;
            int pY = 1;

            ConvolutionParameter parameter = new ConvolutionParameter(height, width, kH, kW, pY, pX, sX, sY);

            // kernel
            // kernel patternA
            INDArray kernel = Nd4j.create(new double[][]{
                    {1, 1},
                    {1, 1}
            });

            //Input data: shape [miniBatch,depth,height,width]
            INDArray input = Nd4j.create(new double[][]{
                    {1, 2},
                    {3, 4}
            });

            INDArray result = ConvolutionUtil.Convolution2D(input, kernel, parameter);

            System.out.println(result);

            INDArray expected = Nd4j.create(new double[][]{
                    {1.0000, 3.0000, 2.0000},
                    {4.0000, 10.0000, 6.0000},
                    {3.0000, 7.0000, 4.0000}
            });
            assertEquals(expected, result);
        }
    }

    @Test
    void convolution3D() {

        int height = 3;
        int width = 3;

        int kH = 2;
        int kW = 2;
        int sX = 1;
        int sY = 1;
        int pX = 0;
        int pY = 0;

        int depth = 2;

        ConvolutionParameter parameter = new ConvolutionParameter(height, width, kH, kW, pY, pX, sX, sY);

        // kernel
        // kernel patternA
        INDArray kernel = Nd4j.create(new double[][][]{
                {
                        {1, 2},
                        {3, 4}
                },
                {
                        {5, 6},
                        {7, 8}
                },
        });

        //Input data: shape [miniBatch,depth,height,width]
        INDArray input = Nd4j.create(new double[][][]{
                {
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9}
                },
                {
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9}
                }
        });

        INDArray result = ConvolutionUtil.Convolution3D(input, kernel, depth, parameter);

        INDArray expected = Nd4j.create(new double[][]{
                {122, 158},
                {230, 266}
        });

        assertEquals(expected, result);
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
        INDArray expected = Nd4j.create(new double[][][]{
                {
                        {189.0000, 216.0000, 243.0000},
                        {324.0000, 351.0000, 378.0000},
                        {459.0000, 486.0000, 513.0000},
                },
                {
                        {189.0000, 216.0000, 243.0000},
                        {324.0000, 351.0000, 378.0000},
                        {459.0000, 486.0000, 513.0000},
                },
        });
        assertEquals(expected, result);
    }
}
