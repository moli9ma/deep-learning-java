package net.moli9ma.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class AffineLayer {

    INDArray weight;
    INDArray bias;

    INDArray dWeight;
    INDArray dBias;

    INDArray x;

    public AffineLayer(INDArray weight, INDArray bias) {
        this.weight = weight;
        this.bias = bias;
    }

    public INDArray forward(INDArray x) {
        this.x = x;
        INDArray out = x.mmul(this.weight).add(this.bias);
        return out;
    }

    public INDArray backward(INDArray dout) {
        INDArray dx = dout.mmul(this.weight.transpose());
        this.dWeight = this.x.mmul(dout);

        this.dBias = dout.sum(0);
        return dx;
    }
}
