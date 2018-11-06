package net.moli9ma.deeplearning.optimizer;

import net.moli9ma.deeplearning.NdUtil;
import net.moli9ma.deeplearning.optimizer.MomentumOptimizer;
import net.moli9ma.deeplearning.optimizer.StochasticGradientDescentOptimizer;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.function.Function;

public class MomentumOptimizerTest {

    @Test
    void t1() {

        Function<INDArray, Double> getGradient = x -> {
            double x1 = x.getDouble(0);
            double x2 = x.getDouble(1);
            double result = Math.pow(x1, 2)/20  + Math.pow(x2, 2);
            return result;
        };

        int iterateNum = 100;

        INDArray parameter = Nd4j.create(new double[]{10.0, 4.0});
        //INDArray parameter = Nd4j.rand(2, 1);
        HashMap<String, INDArray> params = new HashMap<>();
        params.put("w1", parameter);

        MomentumOptimizer optimizer = new MomentumOptimizer();

        List<INDArray> results = new ArrayList<>();

        for (int i = 0; i < iterateNum; i++) {
            INDArray grad = NdUtil.NumericalGradient(getGradient, params.get("w1"));
            HashMap<String, INDArray> grads = new HashMap<>();
            grads.put("w1", grad);
            optimizer.update(params, grads);

            INDArray output = Nd4j.create(new double[]{0,0});
            Nd4j.copy(params.get("w1"), output);
            results.add(output);
        }
    }

}
