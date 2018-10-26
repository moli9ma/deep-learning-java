package net.moli9ma.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;

public class TwolayerNetParameter {

    INDArray weight1;
    INDArray bias1;

    INDArray weight2;
    INDArray bias2;

    /**
     * @param weight1 重み1
     * @param bias1   バイアス1
     * @param weight2 重み2
     * @param bias2   バイアス2
     */
    public TwolayerNetParameter(INDArray weight1, INDArray bias1, INDArray weight2, INDArray bias2) {
        this.weight1 = weight1;
        this.bias1 = bias1;
        this.weight2 = weight2;
        this.bias2 = bias2;
    }
}
