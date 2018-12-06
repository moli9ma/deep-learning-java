package net.moli9ma.deeplearning.layer;

import net.moli9ma.deeplearning.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

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
     *
     * Constructor
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
        INDArray convolved = colInput.mmul(colWeight);
        int miniBatch = (int) x.shape()[0];
        int depth = (int) x.shape()[1];
        return reshape2Image(miniBatch, depth, convolved);
    }


    private INDArray reshape2Image(int miniBatch, int depth, INDArray convolved4D) {
        INDArray out = Nd4j.create(new int[]{miniBatch, depth, convolutionParameter.getOutputWidth(), convolutionParameter.getOutputHeight()}, 'c');
        ColumnIterator columnIterator = new ColumnIterator(convolved4D, convolutionParameter.getKernelWidth(), convolutionParameter.getKernelHeight());
        for (int i = 0; i < miniBatch; i++) {
            for (int j = 0; j < depth; j++) {
                INDArray col2D = columnIterator.next();
                INDArray image2D = col2D.reshape(convolutionParameter.getOutputWidth(), convolutionParameter.getOutputHeight());
                out.put(new INDArrayIndex[]{point(i), point(j), all(), all()}, image2D);
            }
        }
        return out;
    }

    @Override
    public INDArray backward(INDArray dout) {
        int miniBatch = (int) this.input.shape()[0];
        int depth = (int) this.input.shape()[1];
        this.dBias = Nd4j.sum(dout);

        dout = reshape2Col(miniBatch, depth, dout);
        System.out.println("dout");
        System.out.println(dout);

        System.out.println("colInput");
        System.out.println(colInput);

        this.dWeight = this.colInput.mmul(dout);
        System.out.println(this.dBias);
        System.out.println(this.dWeight);

        System.out.println("colWeight");
        System.out.println(this.colWeight);


        INDArray dcol = dout.mmul(this.colWeight.transpose());
        System.out.println(dcol);

        INDArray dx = ConvolutionUtil.Col2Im(miniBatch, depth, dcol, convolutionParameter);
        System.out.println(dx);
        return null;
    }

    private INDArray reshape2Col(int miniBatch, int depth, INDArray image4D) {
        INDArray batchMerged = null;
        for (int i = 0; i < miniBatch; i++) {
            INDArray channelMerged = null;
            for (int j = 0; j < depth; j++) {
                int rows = convolutionParameter.getKernelWidth() * convolutionParameter.getKernelHeight();
                int columns = 1;
                INDArray arr = image4D.get(new INDArrayIndex[]{point(i), point(j)}).reshape(rows, columns);
                channelMerged = (channelMerged == null) ?  arr :  Nd4j.concat(1, channelMerged, arr);
            }
            batchMerged = (batchMerged == null) ?  channelMerged :  Nd4j.concat(0, batchMerged, channelMerged);
        }
        return batchMerged;
    }
}


/*
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中間データ（backward時に使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

        def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        print("col(input)=")
        print(col.shape)
        print(col)

        col_W = self.W.reshape(FN, -1).T
        print("col_W(kernel)=")
        print(col_W.shape)
        print(col_W)

        out = np.dot(col, col_W) + self.b
        print("dotout=")
        print(out)

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

        def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
*/


/*
        [[  148.   296.   444.   592.   148.   296.   444.   592.]
        [  188.   376.   564.   752.   188.   376.   564.   752.]
        [  268.   536.   804.  1072.   268.   536.   804.  1072.]
        [  308.   616.   924.  1232.   308.   616.   924.  1232.]
        [  148.   296.   444.   592.   148.   296.   444.   592.]
        [  188.   376.   564.   752.   188.   376.   564.   752.]
        [  268.   536.   804.  1072.   268.   536.   804.  1072.]
        [  308.   616.   924.  1232.   308.   616.   924.  1232.]]
*/
