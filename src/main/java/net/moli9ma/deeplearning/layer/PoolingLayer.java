package net.moli9ma.deeplearning.layer;

import net.moli9ma.deeplearning.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class PoolingLayer implements Layer {

    private final int poolWidth;
    private final int poolHeight;
    private final int stride;
    private final int padding;

    // 中間データ(逆伝搬で使います)
    private int batchNumber;
    private int channelNumber;
    private int height;
    private int width;
    private INDArray argMax;

    public PoolingLayer(int poolWidth, int poolHeight) {
        this(poolWidth, poolHeight, 1, 0);
    }

    public PoolingLayer(int poolWidth, int poolHeight, int stride, int padding) {
        this.poolWidth = poolWidth;
        this.poolHeight = poolHeight;
        this.stride = stride;
        this.padding = padding;
    }

    @Override
    public INDArray forward(INDArray x) {
        int batchNumber =  (int) x.shape()[0];
        int channelNumber =  (int) x.shape()[1];
        int height = (int) x.shape()[2];
        int width = (int) x.shape()[3];
        this.batchNumber = batchNumber;
        this.channelNumber = channelNumber;
        this.height = height;
        this.width = width;

        int outHeight = 1 +(height - this.poolHeight) / this.stride;
        int outWidth = 1 +(width - this.poolWidth) / this.stride;
        ConvolutionParameter parameter = new ConvolutionParameter(width, height, this.poolWidth, this.poolHeight, padding, padding, stride, stride);

        // Max Pooling
        INDArray out = Nd4j.create(new int[]{batchNumber, channelNumber, outHeight, outWidth});
        argMax = Nd4j.create(new int[]{batchNumber, channelNumber, outHeight, outWidth});
        for (int i = 0; i < batchNumber; i++) {
            for (int j = 0; j < channelNumber; j++) {
                INDArray arr = ConvolutionUtil.Im2col(x.get(new INDArrayIndex[]{point(i), point(j)}), parameter);
                INDArray reshaped = arr.max(0).reshape(outHeight, outWidth);
                out.put(new INDArrayIndex[]{point(i), point(j)}, reshaped);
                INDArray argmax = arr.argMax(0).reshape(outHeight, outWidth);
                argMax.put(new INDArrayIndex[]{point(i), point(j)}, argmax);
            }
        }
        INDArray col = out;
        return col;
    }

    @Override
    public INDArray backward(INDArray x) {
        INDArray out = Nd4j.zeros(new int[]{batchNumber, channelNumber, this.height, this.width});
        for (int i = 0; i < batchNumber; i++) {
            for (int j = 0; j < channelNumber; j++) {

                int maxX = this.width;
                int maxY = this.height;
                int kernelWidth= this.poolWidth;
                int kernelHeight = this.poolHeight;
                WindowIterator iterator = new WindowIterator(
                        maxX,
                        maxY,
                        kernelWidth,
                        kernelHeight,
                        this.stride,
                        this.stride
                );

                for (int k = 0; k < poolHeight; k++) {
                    for (int l = 0; l < poolWidth; l++) {
                        INDArray pool = Nd4j.zeros(new int[]{this.poolHeight * this.poolWidth});
                        double value = x.getDouble(i, j, k, l);
                        long index = this.argMax.getInt(i, j, k, l);
                        pool.put(new INDArrayIndex[]{point(index)}, value);

                        Window empWindow = iterator.next();
                        INDArrayIndex[] indices = new INDArrayIndex[]{
                                NDArrayIndex.interval(empWindow.getStartY(), empWindow.getEndY()),
                                NDArrayIndex.interval(empWindow.getStartX(), empWindow.getEndX())
                        };
                        INDArray emp = Nd4j.create(new int[]{this.height, this.width});
                        emp.put(indices, pool.reshape(this.poolHeight, this.poolWidth));
                        out.get(new INDArrayIndex[]{point(i), point(j)}).addi(emp);
                    }
                }
            }
        }
        return out;
    }
}

/*
class Pooling:
        def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

        def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

        def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx
*/
