package net.moli9ma.deeplearning.layer;

import net.moli9ma.deeplearning.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class SigmoidLayer implements Layer {

    INDArray out;

    @Override
    public INDArray forward(INDArray x) {
        this.out = Transforms.sigmoid(x);
        return out;
    }

    @Override
    public INDArray backward(INDArray dout) {
        return dout.mul(out.rsub(1.0).mul(this.out));
    }
}
