package net.moli9ma.deeplearning.optimizer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;

public class MomentumOptimizer implements Optimizer {

    // Nd4j.zerosLike(origin.get(key))
    public double learningRate;

    public double momentum;

    private HashMap<String, INDArray> v;

    public MomentumOptimizer() {
        this.learningRate = 0.01;
        this.momentum = 0.9;
    }

    public MomentumOptimizer(double learningRate, double momentum) {
        this.learningRate = learningRate;
        this.momentum = momentum;
    }

    public void update(HashMap<String, INDArray> params, HashMap<String, INDArray> grads) {

        if (v == null) {
            v = new HashMap<>();
            for (String key : params.keySet()) {
                v.put(key, Nd4j.zerosLike(params.get(key)));
            }
        }

        for (String key: params.keySet()) {
            this.v.get(key).muli(momentum).subi(grads.get(key).mul(learningRate));
            params.get(key).addi(this.v.get(key));
        }
    }
}
