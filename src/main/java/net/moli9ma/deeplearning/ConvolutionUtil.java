package net.moli9ma.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import javax.naming.CompositeName;

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

    /**
     * 2次元のデータに対してカラム展開を行います。
     *
     * @param input
     * @param convolutionParameter
     * @return
     */
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

    /**
     * 4次元のデータに対してカラム展開を行います。
     *
     * @param input
     * @param convolutionParameter
     * @return
     */
    public static INDArray Im2col4D(INDArray input, ConvolutionParameter convolutionParameter) {
        int batchNumber = (int) input.shape()[0];
        int channelNumber = (int) input.shape()[1];

        INDArray batchMerged = null;
        for (int i = 0; i < batchNumber; i++) {
            INDArray channelMerged = null;
            for (int j = 0; j < channelNumber; j++) {
                INDArray arr = Im2col(input.get(new INDArrayIndex[]{point(i), point(j)}), convolutionParameter);
                channelMerged = (channelMerged == null) ?  arr :  Nd4j.concat(1, channelMerged, arr);
            }
            batchMerged = (batchMerged == null) ?  channelMerged :  Nd4j.concat(0, batchMerged, channelMerged);
        }
        return batchMerged;
    }

    /**
     * kernelをcol展開します。
     * @param kernel
     * @return
     */
    public static INDArray kernel2col4D(INDArray kernel, ConvolutionParameter parameter) {
        int batchNumber = (int) kernel.shape()[0];
        int channelNumber = (int) kernel.shape()[1];
        int colNumber = parameter.getKernelHeight() * parameter.getKernelWidth();

        INDArray batchMerged = null;
        for (int i = 0; i < batchNumber; i++) {
            INDArray channelMerged = null;
            for (int j = 0; j < channelNumber; j++) {
                INDArray arr = kernel.get(new INDArrayIndex[]{point(i), point(j)}).reshape(colNumber, 1);
                channelMerged = (channelMerged == null) ? arr : Nd4j.concat(0, channelMerged, arr);
            }
            batchMerged = (batchMerged == null) ?  channelMerged :  Nd4j.concat(1, batchMerged, channelMerged);
        }
        return batchMerged;
    }

    /**
     * ４次元のカラム展開したデータに対してイメージ復元したデータを返却します。
     *
     * @param batchNumber
     * @param channelNumber
     * @param input
     * @param parameter
     * @return
     */
    public static INDArray Col2Im(final int batchNumber, final int channelNumber, INDArray input, ConvolutionParameter parameter) {

        int maxX = (int) input.shape()[0];
        int maxY = (int) input.shape()[1];
        int kernelWidth = maxX / channelNumber;
        int kernelHeight = maxY / batchNumber;
        int strideX = kernelWidth;
        int strideY = kernelHeight;

        WindowIterator iterator = new WindowIterator(maxX, maxY, kernelWidth, kernelHeight, strideX, strideY);

        INDArray result = null;
        for (Window window : iterator) {
            INDArrayIndex[] indices = new INDArrayIndex[]{
                    NDArrayIndex.interval(window.getStartY(), window.getEndY()),
                    NDArrayIndex.interval(window.getStartX(), window.getEndX())
            };

            INDArray hoge = input.get(indices);
            INDArray x = Col2Im(hoge, parameter);

            result = (result == null) ? x : Nd4j.concat(0, result, x);
        }
        return result.reshape(batchNumber, channelNumber, parameter.getInputWidth(), parameter.getInputHeight());
    }


    /**
     * 2次元のカラム展開したデータに対して、復元したデータを返却します。
     *
     * @param input
     * @param convolutionParameter
     * @return
     */
    public static INDArray Col2Im(INDArray input, ConvolutionParameter convolutionParameter) {

        // colイテレータの初期化
        int maxX = (int) input.shape()[0];
        int maxY = (int) input.shape()[1];
        int kernelWidth = 1;
        int kernelHeight = convolutionParameter.getKernelWidth() * convolutionParameter.getKernelHeight();
        WindowIterator colIterator = new WindowIterator(maxX, maxY, kernelWidth, kernelHeight, 1, 1);

        // 結果となる行列(im)を及びイテレータを初期化する
        // img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
        int imageW = convolutionParameter.getInputWidth() + 2 * convolutionParameter.getPaddingWidth() + convolutionParameter.getStrideY() - 1;
        int imageH = convolutionParameter.getInputHeight() + 2 * convolutionParameter.getPaddingHeight() + convolutionParameter.getStrideX() - 1;

/*
        System.out.println(imageH);
        System.out.println(imageW);
        System.out.println(convolutionParameter.getInputHeight());
        System.out.println(convolutionParameter.getInputWidth());
*/
        INDArray result = Nd4j.zeros(imageW, imageH);

        WindowIterator imIterator = new WindowIterator(
                (int) result.shape()[0],
                (int) result.shape()[1],
                convolutionParameter.getKernelWidth(),
                convolutionParameter.getKernelHeight(),
                convolutionParameter.getStrideX(),
                convolutionParameter.getStrideY()
                );


        // 入力(行列)からkernelで取得されたであろうデータを抜き出して、結果行列に加算する
        for (Window colWindow : colIterator) {

            INDArrayIndex[] indices = new INDArrayIndex[]{
                    NDArrayIndex.interval(colWindow.getStartY(), colWindow.getEndY()),
                    NDArrayIndex.interval(colWindow.getStartX(), colWindow.getEndX())
            };

            INDArray filteredData = input.get(indices).reshape(
                    convolutionParameter.getKernelWidth(),
                    convolutionParameter.getKernelHeight());

            Window imWindow = imIterator.next();
            INDArray emp = Nd4j.zeros(imageW, imageH);
            emp.put(new INDArrayIndex[]{
                    NDArrayIndex.interval(imWindow.getStartY(), imWindow.getEndY()),
                    NDArrayIndex.interval(imWindow.getStartX(), imWindow.getEndX())
            }, filteredData);

/*
            System.out.println(filteredData);
            System.out.println(emp);
*/
            result.addi(emp);
        }
        return result;
    }
}
