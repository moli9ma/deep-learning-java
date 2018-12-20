package net.moli9ma.deeplearning.layer;

import net.moli9ma.deeplearning.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class ConvolutionLayer implements Layer {

    ConvolutionParameter convolutionParameter;

    // 重み・バイアスパラメータ
    public INDArray weight;
    public INDArray bias;

    // 重み・バイアスパラメータの勾配
    public INDArray dWeight;
    public INDArray dBias;

    // 中間データ （backward時に使用）
    INDArray input;
    INDArray colInput;
    INDArray colWeight;

    /**
     * Constructor
     *
     * @param convolutionParameter
     * @param weight
     * @param bias
     */
    public ConvolutionLayer(ConvolutionParameter convolutionParameter, INDArray weight, INDArray bias) {
        this.convolutionParameter = convolutionParameter;
        this.weight = weight;
        this.bias = bias;
    }

    @Override
    public INDArray forward(INDArray x) {
        INDArray colInput = ConvolutionUtil.Im2col4D(x, this.convolutionParameter);
        INDArray colWeight = ConvolutionUtil.kernel2col4D(this.weight, this.convolutionParameter);
        this.input = x;
        this.colInput = colInput;
        this.colWeight = colWeight;
        INDArray convolved = colInput.mmul(colWeight).addRowVector(this.bias);
        int batchNumber = (int) x.shape()[0];
        INDArray reshaped = convolved.reshape(new int[]{batchNumber, convolutionParameter.getOutputHeight(), convolutionParameter.getOutputHeight(), -1});
        INDArray out = reshaped.permute(0, 3, 1, 2);
        return out;
    }

    @Override
    public INDArray backward(INDArray dout) {

        int batchNumber = (int) this.weight.shape()[0];
        int channelNumber = (int) this.weight.shape()[1];
        int height = (int) this.weight.shape()[2];
        int width = (int) this.weight.shape()[3];

        dout = dout.permute(0, 2, 3, 1).reshape(-1, batchNumber);
/*
        System.out.println("dout");
        System.out.println(dout.shapeInfoToString());
*/

        this.dBias = Nd4j.sum(dout, 0);
/*
        System.out.println("dBias");
        System.out.println(this.dBias);
        System.out.println(this.dBias.shapeInfoToString());
                System.out.println("colInput");
        System.out.println(colInput);
*/


        this.dWeight = this.colInput.transpose().mmul(dout);
        this.dWeight = this.dWeight.permute(0, 1).reshape(batchNumber, channelNumber, height, width);

/*
        System.out.println(this.dWeight);
        System.out.println("colWeight");
        System.out.println(this.colWeight);
*/

        INDArray dcol = dout.mmul(this.colWeight.transpose());

/*
        System.out.println("dcol");
        System.out.println(dcol.shapeInfoToString());
*/

        dcol = dcol.reshape(
                this.input.shape()[0],
                convolutionParameter.getOutputHeight(),
                convolutionParameter.getOutputWidth(),
                this.input.shape()[1],
                convolutionParameter.getKernelHeight(),
                convolutionParameter.getKernelWidth()).permute(0, 3, 4, 5, 1, 2);

        INDArray dx = Convolution.col2im(dcol, 1, 1, 0,0, convolutionParameter.getKernelHeight(), convolutionParameter.getKernelWidth());
/*

        System.out.println("dx");
        System.out.println(dx.shapeInfoToString());
*/

        return dx;
    }
}