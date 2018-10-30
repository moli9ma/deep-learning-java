package net.moli9ma.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SigmoidLayer {

    INDArray out;

    public INDArray forward(INDArray x) {
        this.out = NdUtil.Sigmoid(x);
        return out;
    }

    public INDArray backward(INDArray dout) {
        return dout.mul(out.rsub(1.0).mul(this.out));
    }
}
