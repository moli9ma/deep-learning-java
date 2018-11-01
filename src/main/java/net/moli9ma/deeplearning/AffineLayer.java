package net.moli9ma.deeplearning;

import com.sun.org.apache.bcel.internal.generic.LADD;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class AffineLayer implements Layer {

    INDArray weight;
    INDArray bias;

    INDArray dWeight;
    INDArray dBias;

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
