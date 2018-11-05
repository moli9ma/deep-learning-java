package net.moli9ma.deeplearning.optimizer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.HashMap;

public class RMSpropOptimizer implements Optimizer
{

    public double learningRate;

    public double decayRate;

    private HashMap<String, INDArray> h;

    public RMSpropOptimizer() {
        this.learningRate = 0.01;
        this.decayRate = 0.99;
    }

    public RMSpropOptimizer(double learningRate, double decayRate) {
        this.learningRate = learningRate;
        this.decayRate = decayRate;
    }

    @Override
    public void update(HashMap<String, INDArray> params, HashMap<String, INDArray> grads) {

        if (h == null) {
            h = new HashMap<>();
            for (String key : params.keySet()) {
                h.put(key, Nd4j.zerosLike(params.get(key)));
            }
        }

        for (String key: params.keySet()) {

            this.h.get(key).muli(this.decayRate);
            this.h.get(key).addi(Transforms.pow(grads.get(key),2).mul(1 - this.decayRate));

            // 平方根
            INDArray squareRoot = Transforms.sqrt(this.h.get(key)).add(1e-7);
            params.get(key).subi(grads.get(key).mul(learningRate).div(squareRoot));
        }


    }
}

/*
class RMSprop:

        """RMSprop"""

        def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

        def update(self, params, grads):
        if self.h is None:
        self.h = {}
        for key, val in params.items():
        self.h[key] = np.zeros_like(val)

        for key in params.keys():
        self.h[key] *= self.decay_rate
        self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
        params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
*/
