package net.moli9ma.deeplearning.layer;

import net.moli9ma.deeplearning.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;

public class AffineLayer implements Layer {

    public INDArray weight;
    public INDArray bias;

    public INDArray dWeight;
    public INDArray dBias;

    INDArray x;

    public AffineLayer(INDArray weight, INDArray bias) {
        this.weight = weight;
        this.bias = bias;
    }

    @Override
    public INDArray forward(INDArray x) {
        this.x = x;
        INDArray out = x.mmul(this.weight).addRowVector(this.bias);
        return out;
    }

    @Override
    public INDArray backward(INDArray dout) {
        INDArray dx = dout.mmul(this.weight.transpose());
        this.dWeight = x.transpose().mmul(dout);
        this.dBias = dout.sum(0);
        return dx;
    }
}
