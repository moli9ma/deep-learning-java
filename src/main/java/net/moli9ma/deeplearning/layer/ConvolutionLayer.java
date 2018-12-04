package net.moli9ma.deeplearning.layer;

import net.moli9ma.deeplearning.ConvolutionParameter;
import net.moli9ma.deeplearning.ConvolutionUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ConvolutionLayer implements Layer {

    ConvolutionParameter convolutionParameter;

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

        int miniBatch = (int) x.shape()[0];
        int depth = (int) x.shape()[1];
        return ConvolutionUtil.Convolution4D(x, weight, miniBatch, depth, convolutionParameter);
    }

    @Override
    public INDArray backward(INDArray dout) {
        int miniBatch = (int) this.input.shape()[0];
        int depth = (int) this.input.shape()[1];

        System.out.println("dout");
        System.out.println(dout);

        this.dBias = Nd4j.sum(dout);
        dout = dout.reshape(convolutionParameter.getOutputNum(), miniBatch);
        System.out.println("dout");
        System.out.println(dout);

        this.dWeight = this.colInput.mmul(dout);
        System.out.println(this.dBias);
        System.out.println(this.dWeight);


        INDArray dcol = dout.mmul(this.colWeight);
        System.out.println(dcol);

        INDArray dx = ConvolutionUtil.Col2Im(miniBatch, depth, dcol, convolutionParameter);
        System.out.println(dx);
        return null;
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
