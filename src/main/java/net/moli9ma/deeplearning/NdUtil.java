package net.moli9ma.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.function.BiFunction;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;

import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class NdUtil {

    /**
     * sigmoid関数
     *
     * @param input
     * @return
     */
    public static INDArray Sigmoid(INDArray input) {
        return Transforms.exp(input.neg()).add(1.0).rdiv(1.0);
    }

    /**
     * ReLU関数
     *
     * @param input
     * @return
     */
    public static INDArray ReLU(INDArray input) {
        return Transforms.relu(input);
    }

    /**
     * Tahh関数
     * @param input
     * @return
     */
    public  static INDArray Tahh(INDArray input) {
        return Transforms.tan(input);
    }

    /**
     * Softmax
     *
     * @param input
     * @return
     */
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
        long batch_size = input.size(0);
        double delta = 1e-7;
        return -t.mul(Transforms.log(input.add(delta))).sumNumber().doubleValue() / batch_size;
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
     * 引数を2つ取る関数f(x0, x1)に対する勾配を返却します。
     * 勾配とは、各変数に対する偏微分の結果をベクトルとしてまとめたものです。
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

        return Nd4j.create(new double[]{r1,r2});
    }

    /**
     * 関数fに対する勾配を返却します。
     *
     * @param f
     * @param x
     * @return
     */
    public static INDArray NumericalGradient(Function<INDArray, Double> f, INDArray x) {
        long rows = x.size(0);
        long cols = x.size(1);
        double h = 1e-4;
        INDArray grad = Nd4j.zerosLike(x);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                double tmp_val = x.getDouble(r, c);
                x.putScalar(r, c, tmp_val + h);
                double fxh1 = f.apply(x);
                x.putScalar(r, c, tmp_val - h);
                double fxh2 = f.apply(x);
                double g = (fxh1 - fxh2) / (2.0 * h);
                grad.putScalar(r, c, g);
                x.putScalar(r, c, tmp_val);
            }
        return grad;
    }


    /**
     * 勾配下降法で、最小値を探索します。
     *
     * @param function
     * @param init
     * @param learningRate
     * @param stepNumber
     * @return
     */
    public static INDArray GradientDecent(BiFunction<Double, Double, Double> function, INDArray init, double learningRate, int stepNumber) {
        INDArray x = init;
        for (int i = 0; i < stepNumber; i++) {
            INDArray grad = NumericalGradient2D(function, x);
            x = x.sub(grad.mul(learningRate));
        }
        return x;
    }

    /**
     * INDArrayの平均値を求めます
     *
     * @param x
     * @return
     */
    public static double average(INDArray x) {
        return x.sumNumber().doubleValue() / x.length();
    }

}
