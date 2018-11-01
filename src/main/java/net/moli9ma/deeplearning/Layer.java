package net.moli9ma.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;

interface Layer {
    INDArray forward(INDArray x);
    INDArray backward(INDArray x);
}