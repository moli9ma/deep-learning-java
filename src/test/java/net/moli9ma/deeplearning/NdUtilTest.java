package net.moli9ma.deeplearning;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import java.util.function.BiFunction;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class NdUtilTest {


    @Test
    void sigmoid() {
        {
            INDArray input = Nd4j.create(new double[]{-1.0, 1.0, 2.0});
            INDArray result = NdUtil.Sigmoid(input);
            assertEquals(0.2689414322376251, result.getDouble(0));
            assertEquals(0.7310585975646973, result.getDouble(1));
            assertEquals(0.8807970285415649, result.getDouble(2));
        }
    }

    @Test
    void Softmax() {

        {
            INDArray matrixA = Nd4j.create(new double[]{0.3, 2.9, 4.0},new int[]{1,3});
            INDArray result = NdUtil.Softmax(matrixA);
            System.out.println(result);
        }
    }

    @Test
    void CrossEntropyError() {

        {
            INDArray t = Nd4j.create(new double[] {0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
            INDArray y = Nd4j.create(new double[] {0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0});
            assertEquals(0.51082545709933802, NdUtil.CrossEntropyError(y, t), 5e-6);
        }

        {

            INDArray t = Nd4j.create(new double[] {0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
            INDArray y = Nd4j.create(new double[] {0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0});
            assertEquals(2.3025840929945458, NdUtil.CrossEntropyError(y, t), 5e-6);
        }
    }

    @Test
    void NumericalDiff() {

        // y = 0.01x2 + 0.1x
        {

            Function<Double, Double> function = x -> 0.01*Math.pow(x, 2) + 0.1*x;

            double r1 = NdUtil.NumericalDiff(function, 10);
            assertEquals(0.2999999999986347, r1);

            double r2 = NdUtil.NumericalDiff(function, 5);
            assertEquals(0.1999999999990898, r2);

        }

        {
            // f1(x0, x1)=x0^2 + x1^1
            BiFunction<Double, Double, Double> f1 = (x0, x1) -> Math.pow(x0, 2) + Math.pow(x1, 2);

            // f1(x0, x1) x0 = 3.0, x1 = 4.0の時, x0に関する偏微分
            Function<Double, Double> f2 = x -> f1.apply(x, 4.0);
            double r1 = NdUtil.NumericalDiff(f2, 3);
            assertEquals(6.00000000000378, r1);

            // f1(x0, x1) x0 = 3.0, x1 = 4.0の時, x0に関する偏微分
            Function<Double, Double> f3 = x -> f1.apply(3.0, x);
            double r2 = NdUtil.NumericalDiff(f3, 4);
            assertEquals(7.999999999999119, r2);
        }
    }

    @Test
    void NumericalGradient2D() {

        // f1(x0, x1)=x0^2 + x1^1
        BiFunction<Double, Double, Double> function = (x0, x1) -> Math.pow(x0, 2) + Math.pow(x1, 2);

        {
            INDArray input = Nd4j.create(new double[]{3.0, 4.0,});
            INDArray result = NdUtil.NumericalGradient2D(function, input);
            assertEquals(6.0, result.getDouble(0));
            assertEquals(8.0, result.getDouble(1));
        }

        {
            INDArray input = Nd4j.create(new double[]{0.0, 2.0,});
            INDArray result = NdUtil.NumericalGradient2D(function, input);
            assertEquals(0.0, result.getDouble(0));
            assertEquals(4.0, result.getDouble(1));
        }

        {
            INDArray input = Nd4j.create(new double[]{3.0, 0.0,});
            INDArray result = NdUtil.NumericalGradient2D(function, input);
            assertEquals(6.0, result.getDouble(0));
            assertEquals(0.0, result.getDouble(1));
        }
    }


    @Test
    void NumericalGradient() {

        // f1(x0, x1)=x0^2 + x1^1
        Function<INDArray , Double> function = (x) -> Math.pow(x.getDouble(0), 2) + Math.pow(x.getDouble(1), 2);

        {
            INDArray input = Nd4j.create(new double[]{3.0, 0.0,});
            INDArray result = NdUtil.NumericalGradient(function, input);
            System.out.println(result);
        }
    }


    @Test
    void GradientDecent() {

        // f1(x0, x1)=x0^2 + x1^1
        BiFunction<Double, Double, Double> function = (x0, x1) -> Math.pow(x0, 2) + Math.pow(x1, 2);

        {
            final double learningRate = 0.02;
            final int stepNumber = 100;
            INDArray input = Nd4j.create(new double[]{-3.0, 4.0});
            INDArray result = NdUtil.GradientDecent(function, input, learningRate, stepNumber);
            System.out.println(result);
        }
    }

}
