package net.moli9ma.deeplearning;

import org.apache.commons.math3.analysis.function.Add;

public class AddLayer {

    AddLayerData data;

    public AddLayer() {
    }

    public double foraward(AddLayerData data) {
        double out = data.x + data.y;
        return out;
    }

    public AddLayerData backward(double dout) {
        double dx = dout * 1;
        double dy = dout * 1;
        return new AddLayerData(dx, dy);
    }
}
