package net.moli9ma.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class ConvolutionUtil {

    public static void main(String[] args) {


    }

    public static INDArray Convolution4D(INDArray input, INDArray kernel, int miniBatch, int depth, ConvolutionParameter parameter) {

        INDArray result = null;

        for (int i = 0; i < miniBatch; i++) {

            INDArray inputMerged = null;
            INDArray kernelMerged = null;

            for (int j = 0; j < depth; j++) {

                INDArray arr = Im2col(input.get(new INDArrayIndex[]{point(i), point(j)}), parameter);

                if (inputMerged == null) {
                    inputMerged = arr;
                } else {
                    inputMerged = Nd4j.concat(1, inputMerged, arr);
                }

                INDArray ker = kernel.get(new INDArrayIndex[]{point(i), point(j)});
                INDArray reshapedKer = ker.reshape(parameter.getKernelHeight() * parameter.getKernelWidth(), 1);

                if (kernelMerged == null) {
                    kernelMerged = reshapedKer;
                } else {
                    kernelMerged = Nd4j.concat(0, kernelMerged, reshapedKer);
                }
            }

            if (result == null) {
                result = inputMerged.mmul(kernelMerged);
            } else {
                result = Nd4j.concat(1, result, inputMerged.mmul(kernelMerged));
            }
        }
        return result;
    }

    public static INDArray Im2col(INDArray input, ConvolutionParameter convolutionParameter) {
        WindowIterator iterator = new WindowIterator (
                convolutionParameter.getInputWidth(),
                convolutionParameter.getInputHeight(),
                convolutionParameter.getKernelWidth(),
                convolutionParameter.getKernelHeight(),
                convolutionParameter.getStrideX(),
                convolutionParameter.getStrideY()
        );

        INDArray result = null;
        for (Window window : iterator) {
            INDArray x = input.get(
                    NDArrayIndex.interval(window.getStartX(), window.getEndX()),
                    NDArrayIndex.interval(window.getStartY(), window.getEndY())
            );

            if (result == null){
                result = x;
            } else {
                result = Nd4j.concat(0, result, x);
            }
        }

        long rows = convolutionParameter.getOutputNum();
        long cols = convolutionParameter.getKernelWidth() * convolutionParameter.getKernelHeight();

        return result.reshape(rows, cols);
    }

    public static INDArray getPadded(INDArray input, ConvolutionParameter convolutionParameter) {
        if (!convolutionParameter.isPaddingAvailable()) {
            return input;
        } else {
            int[] padWidth = new int[]{
                    convolutionParameter.getPaddingWidth(),
                    convolutionParameter.getPaddingHeight()
            };
            return Nd4j.pad(input, padWidth,  Nd4j.PadMode.CONSTANT);
        }
    }
}
