package net.moli9ma.deeplearning;

public class ConvolutionParameter {
    private int inputWidth;
    private int inputHeight;

    private int kernelWidth;
    private int kernelHeight;

    private int paddingWidth;
    private int paddingHeight;

    private int strideX;
    private int strideY;


    public ConvolutionParameter(int inputWidth, int inputHeight, int kernelWidth, int kernelHeight, int paddingWidth, int paddingHeight, int strideX, int strideY) {
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
        this.paddingWidth = paddingWidth;
        this.paddingHeight = paddingHeight;
        this.strideX = strideX;
        this.strideY = strideY;
    }

    public int getInputWidth() {
        return inputWidth;
    }

    public int getInputHeight() {
        return inputHeight;
    }

    public int getKernelWidth() {
        return kernelWidth;
    }

    public int getKernelHeight() {
        return kernelHeight;
    }

    public int getPaddingWidth() {
        return paddingWidth;
    }

    public int getPaddingHeight() {
        return paddingHeight;
    }

    public int getStrideX() {
        return strideX;
    }

    public int getStrideY() {
        return strideY;
    }

    public int getInputHeightWithPadding() {
        return getInputHeight() + (2 * getPaddingHeight());
    }

    public int getInputWidthWithPadding() {
        return getInputWidth() + (2 * getPaddingWidth());
    }

    public int getOutputHeight() {
        return (this.inputHeight + (2 * this.paddingHeight) - this.kernelHeight) / this.strideY + 1;
    }

    public int getOutputWidth() {
        return (this.inputWidth + (2 * this.paddingWidth) - this.kernelWidth) / this.strideX + 1;
    }

    public int getOutputNum() {
        return getOutputWidth() * getOutputHeight();
    }

    public boolean isPaddingAvailable() {
        return (this.getPaddingWidth() >= 1 && this.getPaddingHeight() >= 1);
    }
}