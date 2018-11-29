package net.moli9ma.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class ConvolutionUtil {

    public static INDArray Convolution2D(INDArray input, INDArray kernel, ConvolutionParameter parameter) {
        INDArray arr = Im2col(input, parameter);
        INDArray convolved = arr.mmul(kernel.reshape(parameter.getKernelHeight() * parameter.getKernelWidth(), 1));
        return convolved.reshape(new int[]{parameter.getOutputWidth(), parameter.getOutputHeight()});
    }

    public static INDArray Convolution3D(INDArray input, INDArray kernel, int depth, ConvolutionParameter parameter) {
        INDArray inputMerged = null;
        INDArray kernelMerged = null;

        for (int i = 0; i < depth; i++) {
            INDArray arr = Im2col(input.get(new INDArrayIndex[]{point(i)}), parameter);
            inputMerged = (inputMerged == null) ?  arr :  Nd4j.concat(1, inputMerged, arr);

            INDArray ker = kernel.get(new INDArrayIndex[]{point(i)});
            INDArray reshapedKer = ker.reshape(parameter.getKernelHeight() * parameter.getKernelWidth(), 1);
            kernelMerged = (kernelMerged == null) ?  reshapedKer : Nd4j.concat(0, kernelMerged, reshapedKer);
        }
        INDArray convolved = inputMerged.mmul(kernelMerged);
        return convolved.reshape(new int[]{parameter.getOutputWidth(), parameter.getOutputHeight()});
    }

    public static INDArray Convolution4D(INDArray input, INDArray kernel, int miniBatch, int depth, ConvolutionParameter parameter) {
        INDArray result = Nd4j.create(new int[]{miniBatch, parameter.getOutputWidth(), parameter.getOutputHeight()}, 'c');

        for (int i = 0; i < miniBatch; i++) {
            INDArray input3D = input.get(new INDArrayIndex[]{point(i)});
            INDArray kernel3D = kernel.get(new INDArrayIndex[]{point(i)});
            INDArray convolved = Convolution3D(input3D, kernel3D, depth, parameter);
            result.put(new INDArrayIndex[]{point(i), all(), all()}, convolved);
        }
        return result;
    }

    public static INDArray Im2col(INDArray input, ConvolutionParameter convolutionParameter) {

        // Padding
        if (convolutionParameter.isPaddingAvailable()) {
            int[][] padWidth = new int[][]{
                    {convolutionParameter.getPaddingHeight(), convolutionParameter.getPaddingHeight()},
                    {convolutionParameter.getPaddingWidth(), convolutionParameter.getPaddingWidth()}
            };
            input = Nd4j.pad(input, padWidth,  Nd4j.PadMode.CONSTANT);
        }

        WindowIterator iterator = new WindowIterator (
                convolutionParameter.getInputWidthWithPadding(),
                convolutionParameter.getInputHeightWithPadding(),
                convolutionParameter.getKernelWidth(),
                convolutionParameter.getKernelHeight(),
                convolutionParameter.getStrideX(),
                convolutionParameter.getStrideY()
        );

        INDArray result = null;
        for (Window window : iterator) {
            INDArray x = input.get(
                    NDArrayIndex.interval(window.getStartY(), window.getEndY()),
                    NDArrayIndex.interval(window.getStartX(), window.getEndX())
                    );
            result = (result == null) ? x : Nd4j.concat(0, result, x);
        }

        long rows = convolutionParameter.getOutputNum();
        long cols = convolutionParameter.getKernelWidth() * convolutionParameter.getKernelHeight();
        return result.reshape(rows, cols);
    }

    public static INDArray Col2Im(INDArray input, ConvolutionParameter convolutionParameter) {

        // イテレータの初期化
        int maxX = (int) input.shape()[0];
        int maxY = (int) input.shape()[1];
        int kernelWidth = 1;
        int kernelHeight = convolutionParameter.getKernelWidth() * convolutionParameter.getKernelHeight();
        WindowIterator iterator = new WindowIterator(maxX, maxY, kernelWidth, kernelHeight, 1, 1);

        // 結果となる行列を初期化する
        // img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
        int imageW = 2 * convolutionParameter.getPaddingWidth() + convolutionParameter.getStrideY();
        int imageH = 2 * convolutionParameter.getPaddingHeight() + convolutionParameter.getStrideX();
        INDArray result = Nd4j.zeros(imageW, imageH);


        // 入力(行列)からkernelで取得されたであろうデータを抜き出して、結果行列に加算する
        for (Window window : iterator) {

            INDArrayIndex[] indices = new INDArrayIndex[]{
                    NDArrayIndex.interval(window.getStartY(), window.getEndY()),
                    NDArrayIndex.interval(window.getStartX(), window.getEndX())
            };

            INDArray x = input.get(indices);

            INDArray reshaped = x.reshape(convolutionParameter.getKernelWidth(), convolutionParameter.getKernelHeight());

            x.put(indices, reshaped);
        }

        return result;
    }
}
