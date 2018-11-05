package net.moli9ma.deeplearning.optimizer;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;

public interface Optimizer {
    void update(HashMap<String, INDArray> params, HashMap<String, INDArray> grads);
}
