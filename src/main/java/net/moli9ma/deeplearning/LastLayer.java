package net.moli9ma.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;

interface LastLayer {

    double forward(INDArray x, INDArray t);
    INDArray backward();

}
