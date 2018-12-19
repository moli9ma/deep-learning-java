package net.moli9ma.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;

public interface Network {

    INDArray predict(INDArray x);

    double loss(INDArray input, INDArray t);

    double accuracy(INDArray x, INDArray t);

    HashMap<String, INDArray> numericalGradient(INDArray x, INDArray t);

    HashMap<String, INDArray> gradient(INDArray x, INDArray t);

    HashMap<String, INDArray> getParams();

}