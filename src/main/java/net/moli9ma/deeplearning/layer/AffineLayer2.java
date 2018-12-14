package net.moli9ma.deeplearning.layer;

import org.nd4j.linalg.api.ndarray.INDArray;

public class AffineLayer2 implements Layer {

    public INDArray weight;
    public INDArray bias;

    public INDArray dWeight;
    public INDArray dBias;

    INDArray x;
    long[] originalXShape;

    public AffineLayer2(INDArray weight, INDArray bias) {
        this.weight = weight;
        this.bias = bias;
    }

    @Override
    public INDArray forward(INDArray x) {
        this.originalXShape = x.shape();
        x = x.reshape(x.shape()[0], -1);
        this.x = x;
        INDArray out = x.mmul(this.weight).addRowVector(this.bias);
        return out;
    }

    @Override
    public INDArray backward(INDArray dout) {
        INDArray dx = dout.mmul(this.weight);
        this.dWeight = this.x.mmul(dout);
        this.dBias = dout.sum(0);

        dx = dx.reshape(this.originalXShape);
        return dx;
    }
}
