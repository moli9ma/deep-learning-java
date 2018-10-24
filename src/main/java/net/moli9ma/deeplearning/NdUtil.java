package net.moli9ma.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.function.BiFunction;
import java.util.function.Function;

public class NdUtil {

    public static INDArray Softmax(INDArray input) {
        double sum = Transforms.exp(input).sumNumber().doubleValue();
        return Transforms.exp(input).div(sum);
    }


    /**
     * 正解ラベル(t)に対する入力(input)の交差エントロピー誤差を求めます。
     * 推定確率値が高ければ0に近い値が返却されます。
     *
     * @param input 確率  [0.1, 0.4, 0.6]
     * @param t 正解ラベル(one-hot) [0, 1, 0]
     * @return
     */
    public static double CrossEntropyError(INDArray input, INDArray t) {
        double delta = 1e-7;
        return -t.mul(Transforms.log(input.add(delta))).sumNumber().doubleValue();
    }


    /**
     * 関数f(x)に関しての数値微分を求めます。
     *
     *
     * @param function
     * @param x
     * @return
     */
    public static double NumericalDiff(Function<Double, Double> function, double x) {
        double h = 1e-4;
        return (function.apply(x+h) - function.apply(x-h)) / (2*h);
    }


    /**
     *
     *
     * @param function
     * @param input
     * @return
     */
    public static INDArray NumericalGradient2D(BiFunction<Double, Double, Double> function, INDArray input) {
        double x0 = input.getDouble(0);
        double x1 = input.getDouble(1);

        Function<Double, Double> function4x0 = x -> function.apply(x, x1);
        double r1 = NdUtil.NumericalDiff(function4x0, x0);

        Function<Double, Double> function4x1 = x -> function.apply(x0, x);
        double r2 = NdUtil.NumericalDiff(function4x1, x1);

        return Nd4j.create(new double[]{
                r1,r2
        });
    }
}
