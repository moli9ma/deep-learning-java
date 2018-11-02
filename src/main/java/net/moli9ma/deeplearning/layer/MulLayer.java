package net.moli9ma.deeplearning.layer;

public class MulLayer {

    MulLayerData data;

    public MulLayer() {

    }

    public double forward(MulLayerData data) {
        this.data = data;
        return data.x * data.y;
    }

    public MulLayerData backward(double dout) {
        double dx = dout * this.data.y;
        double dy = dout * this.data.x;
        return new MulLayerData(dx, dy);
    }
}


