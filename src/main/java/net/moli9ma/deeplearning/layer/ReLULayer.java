package net.moli9ma.deeplearning.layer;

import net.moli9ma.deeplearning.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class ReLULayer implements Layer {

    // 入力の要素が0以下ならばTrue, それ以外(0より大きい要素)をFalseとして保持する
    INDArray mask;

    public ReLULayer() { }

/*
    @Override
    public INDArray forward(INDArray x) {

        // create mask array
        mask = Nd4j.create(x.shape());
        Nd4j.copy(x, mask);
        BooleanIndexing.replaceWhere(mask, 0.0, Conditions.lessThan(0.0));
        BooleanIndexing.replaceWhere(mask, 1.0, Conditions.greaterThan(0.0));

        return x.mul(mask);
    }
*/

/*
    @Override
    public INDArray backward(INDArray x) {
        return x.mul(mask);
    }
*/

    @Override
    public INDArray forward(INDArray x) {
        // 要素の値＞0.0の時は1、それ以外の時は0をmaskに格納します。
        // "gt"は"greater than"の意味です。
        this.mask = x.gt(0.0);
        return Transforms.relu(x);
    }

    @Override
    public INDArray backward(INDArray dout) {
        return dout.mul(mask);
    }

}
