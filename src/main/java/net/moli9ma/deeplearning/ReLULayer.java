package net.moli9ma.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

public class ReLULayer {

    // 入力の要素が0以下ならばTrue, それ以外(0より大きい要素)をFalseとして保持する
    INDArray mask;

    public ReLULayer() { }

    public INDArray forward(INDArray x) {

        // create mask array
        mask = Nd4j.create(x.shape());
        Nd4j.copy(x, mask);
        BooleanIndexing.replaceWhere(mask, 0.0, Conditions.lessThan(0.0));
        BooleanIndexing.replaceWhere(mask, 1.0, Conditions.greaterThan(0.0));

        return x.mul(mask);
    }

    public INDArray backward(INDArray x) {
        return x.mul(mask);
    }
}
