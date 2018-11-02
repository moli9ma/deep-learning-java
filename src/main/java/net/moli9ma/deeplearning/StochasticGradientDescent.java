package net.moli9ma.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;

public class StochasticGradientDescent {

    double learningRate;

    public StochasticGradientDescent() {
        this.learningRate = 0.01;
    }

    public StochasticGradientDescent(double learningRate) {
        this.learningRate = learningRate;
    }

    public void update(TwolayerNetParameter params, TwolayerNetParameter grads) {
        params.weight1.subi(grads.weight1.mul(this.learningRate));
        params.bias1.subi(grads.bias1.mul(this.learningRate));
        params.weight2.subi(grads.weight2.mul(this.learningRate));
        params.bias2.subi(grads.bias2.mul(this.learningRate));
    }

    public void update(HashMap<String, INDArray> params, HashMap<String, INDArray> grads) {
        for (String key : params.keySet()) {
            params.get(key).subi(grads.get(key).mul(this.learningRate));
        }
    }
}
