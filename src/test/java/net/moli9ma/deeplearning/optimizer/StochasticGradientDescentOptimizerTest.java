package net.moli9ma.deeplearning.optimizer;

import net.moli9ma.deeplearning.TwolayerNetParameter;
import net.moli9ma.deeplearning.optimizer.StochasticGradientDescentOptimizer;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class StochasticGradientDescentOptimizerTest {

    @Test
    void t1() {
        TwolayerNetParameter params = new TwolayerNetParameter(
                Nd4j.ones(2, 2),
                Nd4j.ones(2, 2),
                Nd4j.ones(2, 2),
                Nd4j.ones(2, 2)
        );

        TwolayerNetParameter grads = new TwolayerNetParameter(
                Nd4j.ones(2, 2),
                Nd4j.ones(2, 2),
                Nd4j.ones(2, 2),
                Nd4j.ones(2, 2)
        );

        StochasticGradientDescentOptimizer sgd =  new StochasticGradientDescentOptimizer();
        sgd.update(params, grads);

        assertEquals("[[    0.9900,    0.9900], \n" +
                " [    0.9900,    0.9900]]", params.weight1.toString());
        assertEquals("[[    0.9900,    0.9900], \n" +
                " [    0.9900,    0.9900]]", params.weight2.toString());
        assertEquals("[[    0.9900,    0.9900], \n" +
                " [    0.9900,    0.9900]]", params.bias1.toString());
        assertEquals("[[    0.9900,    0.9900], \n" +
                " [    0.9900,    0.9900]]", params.bias2.toString());
    }

    @Test
    void t2() {

        HashMap<String, INDArray> params = new HashMap<>();
        params.put("weight1", Nd4j.ones(2, 2));
        params.put("weight2", Nd4j.ones(2, 2));
        params.put("bias1", Nd4j.ones(2, 2));
        params.put("bias2", Nd4j.ones(2, 2));

        HashMap<String, INDArray> grads = new HashMap<>();
        grads.put("weight1", Nd4j.ones(2, 2));
        grads.put("weight2", Nd4j.ones(2, 2));
        grads.put("bias1", Nd4j.ones(2, 2));
        grads.put("bias2", Nd4j.ones(2, 2));

        StochasticGradientDescentOptimizer sgd =  new StochasticGradientDescentOptimizer();
        sgd.update(params, grads);

        assertEquals("[[    0.9900,    0.9900], \n" +
                " [    0.9900,    0.9900]]", params.get("weight1").toString());
        assertEquals("[[    0.9900,    0.9900], \n" +
                " [    0.9900,    0.9900]]", params.get("weight2").toString());
        assertEquals("[[    0.9900,    0.9900], \n" +
                " [    0.9900,    0.9900]]", params.get("bias1").toString());
        assertEquals("[[    0.9900,    0.9900], \n" +
                " [    0.9900,    0.9900]]", params.get("bias2").toString());
    }
}
