package net.moli9ma.deeplearning.layer;

import net.moli9ma.deeplearning.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

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
        int batchNumber = (int) x.shape()[0];
        int channelNumber = (int) x.shape()[1];
        int height = (int) x.shape()[2];
        int width = (int) x.shape()[3];
        this.batchNumber = batchNumber;
        this.channelNumber = channelNumber;
        this.height = height;
        this.width = width;

        int outHeight = 1 + (height - this.poolHeight) / this.stride;
        int outWidth = 1 + (width - this.poolWidth) / this.stride;
        ConvolutionParameter parameter = new ConvolutionParameter(width, height, this.poolWidth, this.poolHeight, padding, padding, stride, stride);
        INDArray col = ConvolutionUtil.Im2col4D(x, parameter);
        System.out.println("col.shape");
        System.out.println(col.shapeInfoToString());

        col = col.reshape(new int[]{-1, this.poolHeight * this.poolWidth});

        INDArray out = col.max(1);
        out = out.reshape(batchNumber, outHeight, outWidth, channelNumber);
        out = out.permute(0, 3, 1, 2);
        this.argMax = col.argMax(1);
        return out;
    }

    @Override
    public INDArray backward(INDArray x) {
        System.out.println("x : ");
        System.out.println(x);
        x = x.permute(0, 2, 3, 1);
        System.out.println("x : ");
        System.out.println(x.shapeInfoToString());

        System.out.println("x flat: ");

        INDArray flatten = x.reshape(this.argMax.size(0), 1);

        System.out.println("arg max: ");
        //System.out.println(this.argMax);
        System.out.println(this.argMax.shapeInfoToString());

        int poolSize = this.poolWidth * this.poolHeight;
        INDArray dmax = Nd4j.zeros(this.argMax.size(0), poolSize);
        for (int i = 0; i < this.argMax.size(0); i++) {
            long idx = (long) this.argMax.getDouble(i);
            dmax.put(new INDArrayIndex[]{point(i), point(idx)}, flatten.getDouble(i));
        }

        System.out.println("dmax: ");
        System.out.println(dmax.shapeInfoToString());
        System.out.println(dmax);

        // col2Image
        INDArray out = Nd4j.zeros(new int[]{batchNumber, channelNumber, this.height, this.width});

        for (int i = 0; i < batchNumber; i++) {
            for (int j = 0; j < channelNumber; j++) {

                int z = 0;
                INDArray image = Nd4j.create(new int[]{this.height, this.width});
                int maxX = this.width;
                int maxY = this.height;
                int kernelWidth = this.poolWidth;
                int kernelHeight = this.poolHeight;
                WindowIterator imageIterator = new WindowIterator(
                        maxX,
                        maxY,
                        kernelWidth,
                        kernelHeight,
                        this.stride,
                        this.stride
                );

                for (Window imageWindow : imageIterator) {
                    INDArrayIndex[] indices = new INDArrayIndex[]{
                            NDArrayIndex.interval(imageWindow.getStartY(), imageWindow.getEndY()),
                            NDArrayIndex.interval(imageWindow.getStartX(), imageWindow.getEndX())
                    };
                    INDArray pooledArray = dmax.get(new INDArrayIndex[]{point(z)});
                    image.get(indices).addi(pooledArray.reshape(this.poolHeight, this.poolWidth));
                    z++;
                }
                out.put(new INDArrayIndex[]{point(i), point(j)}, image);
            }
        }

        System.out.println("out");
        System.out.println(out.shapeInfoToString());

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
