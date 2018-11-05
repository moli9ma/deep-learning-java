package net.moli9ma.deeplearning.optimizer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.HashMap;

public class AdamOptimizer implements Optimizer
{

    public double learningRate;
    public double beta1;
    public double beta2;
    public double iterate;


    private HashMap<String, INDArray> m;
    private HashMap<String, INDArray> v;


    @Override
    public void update(HashMap<String, INDArray> params, HashMap<String, INDArray> grads) {

        if (m == null) {
            m = new HashMap<>();
            v = new HashMap<>();

            for (String key : params.keySet()) {
                m.put(key, Nd4j.zerosLike(params.get(key)));
                v.put(key, Nd4j.zerosLike(params.get(key)));
            }
        }

        this.iterate += 1;
        double lrT = this.learningRate * Math.sqrt(1.0 - Math.pow(this.beta2, this.iterate)) / Math.sqrt(1.0 - Math.pow(this.beta1, this.iterate));

        for (String key: params.keySet()) {
            this.m.get(key).addi(grads.get(key).sub(this.m.get(key)).mul((1 - this.beta1)));
            this.v.get(key).addi(Transforms.pow(grads.get(key), 2).sub(this.v.get(key)).mul(1 - this.beta2));

            INDArray squareRoot = Transforms.sqrt(this.v.get(key)).add(1e-7);
            params.get(key).subi(grads.get(key).mul(lrT).div(squareRoot));
        }
    }
}


/*
class Adam:

        """Adam (http://arxiv.org/abs/1412.6980v8)"""

        def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

        def update(self, params, grads):
        if self.m is None:
        self.m, self.v = {}, {}
        for key, val in params.items():
        self.m[key] = np.zeros_like(val)
        self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
        #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
        #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
        self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
        self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

        params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

        #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
        #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
        #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
*/
