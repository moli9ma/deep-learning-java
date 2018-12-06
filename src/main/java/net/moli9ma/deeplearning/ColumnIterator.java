package net.moli9ma.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Iterator;

/**
 * カラム展開した4次元データから2次元データ毎にイテレーションします。
 *
 *
 */
public class ColumnIterator implements Iterable<INDArray>, Iterator<INDArray> {

    private final INDArray column;
    private final int kernelWidth;
    private final int kernelHeight;
    private final WindowIterator windowIterator;

    public ColumnIterator(INDArray column, int kernelWidth, int kernelHeight) {
        this.column = column;
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;

        int maxX = (int) column.shape()[1];
        int maxY = (int) column.shape()[0];
        int width = 1;
        int height = kernelWidth * kernelHeight;
        int strideX = width;
        int strideY = height;
        this.windowIterator = new WindowIterator(maxX, maxY, width, height, strideX, strideY);
    }

    @Override
    public Iterator<INDArray> iterator() {
        return this;
    }

    @Override
    public boolean hasNext() {
        return this.windowIterator.hasNext();
    }

    @Override
    public INDArray next() {
        Window window = this.windowIterator.next();
        INDArrayIndex[] indices = new INDArrayIndex[]{
                NDArrayIndex.interval(window.getStartY(), window.getEndY()),
                NDArrayIndex.interval(window.getStartX(), window.getEndX())
        };
        return this.column.get(indices);
    }
}